#!/usr/bin/env python3
"""
Combined load test — hits /upload/ and /chat/ CONCURRENTLY in the same run,
sharing one clock, so you get a realistic mixed-traffic picture instead of
testing each endpoint in isolation one after another.

Usage:
    python load_testing.py --base-url http://localhost:8000 \
        --session-id <existing-session-for-chat> \
        --pdf sample.pdf \
        --chat-requests 50 --chat-concurrency 10 \
        --upload-requests 10 --upload-concurrency 5 \
        --output-chat chat_combined.json --output-upload upload_combined.json

    # Chat only (skip upload traffic)
    python load_testing.py --base-url http://localhost:8000 \
        --session-id <id> --chat-requests 50 --chat-concurrency 10 --skip-upload

    # Upload only (skip chat traffic)
    python load_testing.py --base-url http://localhost:8000 \
        --pdf sample.pdf --upload-requests 10 --upload-concurrency 5 --skip-chat

Both output JSON files share the same test_start clock, so if you load
BOTH into the dashboard (Run A = chat, Run B = upload) the timeline chart
shows real concurrent traffic on one shared time axis.

Requires: pip install aiohttp
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

DEFAULT_QUERIES = [
    "What is the payment term in this contract?",
    "What is the governing law?",
    "What is the notice period for termination?",
    "Summarize this document.",
    "Explain the force majeure clause.",
]


# ---------------------------------------------------------------------------
# Chat traffic
# ---------------------------------------------------------------------------

async def send_chat(session, base_url, session_id, query, timeout, verbose, idx, test_start):
    url = f"{base_url.rstrip('/')}/chat/"
    payload = {"query": query, "session_id": session_id, "reset_session": False}
    start = time.perf_counter()
    offset = start - test_start
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            body = await resp.json()
            latency = time.perf_counter() - start
            success = resp.status == 200 and bool(body.get("answer"))
            record = {
                "idx": idx, "offset": offset, "success": success, "latency": latency,
                "status": resp.status, "intent": body.get("intent"),
                "llm_latency_ms": body.get("llm_latency_ms"),
                "input_tokens": body.get("input_tokens"), "output_tokens": body.get("output_tokens"),
                "query": query, "detail": "" if success else str(body)[:200],
            }
    except Exception as e:
        latency = time.perf_counter() - start
        detail = str(e) or f"{type(e).__name__} (likely timed out after {timeout}s)"
        record = {"idx": idx, "offset": offset, "success": False, "latency": latency, "status": None,
                   "intent": None, "llm_latency_ms": None, "input_tokens": None, "output_tokens": None,
                   "query": query, "detail": detail}
    if verbose:
        status = "OK" if record["success"] else "FAIL"
        print(f"  [chat]   #{idx:<3} {status:<4} {latency:6.2f}s  intent={record['intent']}  {record['detail'][:60]}")
    return record


async def run_chat_traffic(session, base_url, session_id, total_requests, concurrency,
                            queries, timeout, verbose, test_start):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(idx, query):
        async with semaphore:
            results.append(await send_chat(session, base_url, session_id, query, timeout, verbose, idx, test_start))

    tasks = [asyncio.create_task(bounded(i + 1, queries[i % len(queries)])) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# Upload traffic
# ---------------------------------------------------------------------------

async def send_upload(session, base_url, pdf_path, timeout, verbose, idx, test_start):
    url = f"{base_url.rstrip('/')}/upload/"
    data = aiohttp.FormData()
    data.add_field("files", pdf_path.read_bytes(), filename=pdf_path.name, content_type="application/pdf")
    start = time.perf_counter()
    offset = start - test_start
    try:
        async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            body = await resp.json()
            latency = time.perf_counter() - start
            success = resp.status == 200 and body.get("uploaded_count", 0) > 0
            record = {
                "idx": idx, "offset": offset, "success": success, "latency": latency,
                "status": resp.status, "session_id": body.get("session_id") if success else None,
                "chunk_count": body.get("files", [{}])[0].get("chunk_count") if success and body.get("files") else None,
                "page_count": body.get("files", [{}])[0].get("page_count") if success and body.get("files") else None,
                "detail": "" if success else str(body)[:200], "filename": pdf_path.name,
            }
    except Exception as e:
        latency = time.perf_counter() - start
        detail = str(e) or f"{type(e).__name__} (likely timed out after {timeout}s)"
        record = {"idx": idx, "offset": offset, "success": False, "latency": latency, "status": None,
                   "session_id": None, "chunk_count": None, "page_count": None, "detail": detail,
                   "filename": pdf_path.name}
    if verbose:
        status = "OK" if record["success"] else "FAIL"
        print(f"  [upload] #{idx:<3} {status:<4} {latency:6.2f}s  file={record['filename']}  {record['detail'][:60]}")
    return record


async def run_upload_traffic(session, base_url, pdf_paths, total_requests, concurrency, timeout, verbose, test_start):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(idx, pdf_path):
        async with semaphore:
            results.append(await send_upload(session, base_url, pdf_path, timeout, verbose, idx, test_start))

    tasks = [asyncio.create_task(bounded(i + 1, pdf_paths[i % len(pdf_paths)])) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def main_async(args):
    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [l.strip() for l in args.queries_file.read_text().splitlines() if l.strip()]

    pdf_paths = []
    if not args.skip_upload:
        if args.pdf_dir:
            pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
        elif args.pdf:
            pdf_paths = [args.pdf]

    connector = aiohttp.TCPConnector(limit=args.chat_concurrency + args.upload_concurrency)
    test_start = time.perf_counter()
    chat_results, upload_results = [], []

    async with aiohttp.ClientSession(connector=connector) as session:
        coros = []
        if not args.skip_chat:
            if not args.session_id:
                print("No --session-id given for chat traffic; skipping chat load.")
            else:
                coros.append(run_chat_traffic(
                    session, args.base_url, args.session_id, args.chat_requests,
                    args.chat_concurrency, queries, args.timeout, args.verbose, test_start,
                ))
        if not args.skip_upload:
            if not pdf_paths:
                print("No --pdf or --pdf-dir given for upload traffic; skipping upload load.")
            else:
                coros.append(run_upload_traffic(
                    session, args.base_url, pdf_paths, args.upload_requests,
                    args.upload_concurrency, args.timeout, args.verbose, test_start,
                ))

        print(f"\nStarting concurrent load: {len(coros)} traffic generator(s) running at once...\n")
        gathered = await asyncio.gather(*coros)

    idx = 0
    if not args.skip_chat and args.session_id:
        chat_results = gathered[idx]; idx += 1
    if not args.skip_upload and pdf_paths:
        upload_results = gathered[idx]; idx += 1

    total_time = time.perf_counter() - test_start
    return chat_results, upload_results, total_time


def print_summary(name, results, total_time):
    if not results:
        return
    successes = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in results]
    print(f"\n--- {name.upper()} ---")
    print(f"  Requests:    {len(results)}")
    print(f"  Successful:  {len(successes)}")
    print(f"  Failed:      {len(results) - len(successes)}")
    print(f"  Throughput:  {len(results)/total_time:.2f} req/s")
    if latencies:
        print(f"  Avg latency: {statistics.mean(latencies):.2f}s")
        print(f"  Median:      {statistics.median(latencies):.2f}s")
        print(f"  Min/Max:     {min(latencies):.2f}s / {max(latencies):.2f}s")


def save_json(path, endpoint, base_url, results, total_requests, concurrency, total_time, extra=None):
    payload = {
        "endpoint": endpoint, "base_url": base_url, "total_requests": total_requests,
        "concurrency": concurrency, "total_time": total_time, "results": results,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {endpoint} results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run chat + upload load concurrently against the same API")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--session-id", help="Existing session_id to use for chat traffic")
    parser.add_argument("--pdf", type=Path, help="Single PDF for upload traffic")
    parser.add_argument("--pdf-dir", type=Path, help="Directory of PDFs to round-robin for upload traffic")
    parser.add_argument("--chat-requests", type=int, default=30)
    parser.add_argument("--chat-concurrency", type=int, default=10)
    parser.add_argument("--upload-requests", type=int, default=10)
    parser.add_argument("--upload-concurrency", type=int, default=5)
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-chat", type=Path, default=Path("chat_combined.json"))
    parser.add_argument("--output-upload", type=Path, default=Path("upload_combined.json"))
    args = parser.parse_args()

    if not args.skip_chat and not args.session_id:
        print("Warning: no --session-id given, chat traffic will be skipped.")
    if not args.skip_upload and not args.pdf and not args.pdf_dir:
        print("Warning: no --pdf/--pdf-dir given, upload traffic will be skipped.")

    print(f"Target: {args.base_url}")
    print(f"Chat:   requests={args.chat_requests} concurrency={args.chat_concurrency} (skip={args.skip_chat})")
    print(f"Upload: requests={args.upload_requests} concurrency={args.upload_concurrency} (skip={args.skip_upload})")

    chat_results, upload_results, total_time = asyncio.run(main_async(args))

    print("\n" + "=" * 55)
    print("COMBINED LOAD TEST REPORT")
    print("=" * 55)
    print(f"Total wall time (both running together): {total_time:.2f}s")
    print_summary("chat", chat_results, total_time)
    print_summary("upload", upload_results, total_time)
    print("=" * 55)

    if chat_results:
        save_json(args.output_chat, "chat", args.base_url, chat_results,
                   args.chat_requests, args.chat_concurrency, total_time)
    if upload_results:
        save_json(args.output_upload, "upload", args.base_url, upload_results,
                   args.upload_requests, args.upload_concurrency, total_time)

    if chat_results and upload_results:
        print("\nLoad both files into the dashboard (Run A = chat, Run B = upload) to see")
        print("the mixed traffic on one shared timeline.")


if __name__ == "__main__":
    main()
