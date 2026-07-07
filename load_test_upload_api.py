#!/usr/bin/env python3
"""
Standalone load test for POST /upload/ only.

Isolated single-endpoint variant of load_testing.py's upload traffic
generator — use this when you want to characterize document-ingestion
throughput/latency on its own, without concurrent chat traffic competing
for OpenAI/Pinecone rate limits.

Usage:
    python load_test_upload_api.py --base-url http://localhost:8000 \
        --pdf sample.pdf --requests 10 --concurrency 5

    # Round-robin a folder of PDFs instead of repeating one file
    python load_test_upload_api.py --base-url http://localhost:8000 \
        --pdf-dir ./sample_pdfs --requests 20 --concurrency 5

Output JSON uses the same schema as load_testing.py's upload results, so
files from either script can be loaded into the same dashboard/analysis.

Requires: pip install aiohttp
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


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
            files = body.get("files") or [{}]
            record = {
                "idx": idx, "offset": offset, "success": success, "latency": latency,
                "status": resp.status, "session_id": body.get("session_id") if success else None,
                "chunk_count": files[0].get("chunk_count") if success else None,
                "page_count": files[0].get("page_count") if success else None,
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
        print(f"#{idx:<3} {status:<4} {latency:6.2f}s  file={record['filename']}  {record['detail'][:60]}")
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


async def main_async(args):
    if args.pdf_dir:
        pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    elif args.pdf:
        pdf_paths = [args.pdf]
    else:
        pdf_paths = []
    if not pdf_paths:
        raise SystemExit("No PDF(s) to upload — pass --pdf or --pdf-dir.")

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    test_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await run_upload_traffic(
            session, args.base_url, pdf_paths, args.requests, args.concurrency,
            args.timeout, args.verbose, test_start,
        )
    total_time = time.perf_counter() - test_start
    return results, total_time


def print_summary(results, total_time):
    successes = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in results]
    print("\n" + "=" * 55)
    print("UPLOAD LOAD TEST REPORT")
    print("=" * 55)
    print(f"  Requests:    {len(results)}")
    print(f"  Successful:  {len(successes)}")
    print(f"  Failed:      {len(results) - len(successes)}")
    print(f"  Total time:  {total_time:.2f}s")
    print(f"  Throughput:  {len(results) / total_time:.2f} req/s")
    if latencies:
        print(f"  Avg latency: {statistics.mean(latencies):.2f}s")
        print(f"  Median:      {statistics.median(latencies):.2f}s")
        print(f"  Min/Max:     {min(latencies):.2f}s / {max(latencies):.2f}s")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Load test the /upload/ endpoint in isolation")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pdf", type=Path, help="Single PDF to upload repeatedly")
    parser.add_argument("--pdf-dir", type=Path, help="Directory of PDFs to round-robin")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("upload_results.json"))
    args = parser.parse_args()

    print(f"Target: {args.base_url}/upload/")
    print(f"Requests={args.requests} concurrency={args.concurrency}")

    results, total_time = asyncio.run(main_async(args))
    print_summary(results, total_time)

    payload = {
        "endpoint": "upload", "base_url": args.base_url, "total_requests": args.requests,
        "concurrency": args.concurrency, "total_time": total_time, "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
