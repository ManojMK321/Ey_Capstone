#!/usr/bin/env python3
"""
Standalone load test for POST /chat/ — multi-step / comparison / compliance
queries that IntentDetector routes to AgenticRAG (see the AgenticRAG
examples in src/agents/intent_detection.py's prompt).

AgenticRAG runs several sequential LLM calls per request (query analysis,
task routing, a specialist agent, validation, final response generation —
see src/agents/agentic_rag.py), so it has a very different cost/latency
profile than the plain KnowledgeRAG path exercised by load_test_chat_api.py.
Load-testing it separately is the only way to see its latency/error/token
behavior without light chat traffic diluting the numbers.

Usage:
    python load_test_rag_api.py --base-url http://localhost:8000 \
        --session-id <existing-session-id> --requests 20 --concurrency 5

Requires: pip install aiohttp
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

DEFAULT_QUERIES = [
    "Compare payment terms across contracts.",
    "Compare termination clauses.",
    "Which contracts expire within 90 days?",
    "Which contracts have unlimited liability?",
    "Find contracts expiring next month and summarize renewal clauses.",
    "Is this contract compliant with procurement policy?",
]


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
                "intent_confidence": body.get("intent_confidence"),
                "llm_latency_ms": body.get("llm_latency_ms"),
                "input_tokens": body.get("input_tokens"), "output_tokens": body.get("output_tokens"),
                "sources": len(body.get("sources") or []),
                "query": query, "detail": "" if success else str(body)[:200],
            }
    except Exception as e:
        latency = time.perf_counter() - start
        detail = str(e) or f"{type(e).__name__} (likely timed out after {timeout}s)"
        record = {"idx": idx, "offset": offset, "success": False, "latency": latency, "status": None,
                   "intent": None, "intent_confidence": None, "llm_latency_ms": None,
                   "input_tokens": None, "output_tokens": None, "sources": None,
                   "query": query, "detail": detail}
    if verbose:
        status = "OK" if record["success"] else "FAIL"
        print(f"#{idx:<3} {status:<4} {latency:6.2f}s  intent={record['intent']}  {record['detail'][:60]}")
    return record


async def run_chat_traffic(session, base_url, session_id, total_requests, concurrency, queries, timeout, verbose, test_start):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(idx, query):
        async with semaphore:
            results.append(await send_chat(session, base_url, session_id, query, timeout, verbose, idx, test_start))

    tasks = [asyncio.create_task(bounded(i + 1, queries[i % len(queries)])) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    return results


async def main_async(args, queries):
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    test_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await run_chat_traffic(
            session, args.base_url, args.session_id, args.requests,
            args.concurrency, queries, args.timeout, args.verbose, test_start,
        )
    total_time = time.perf_counter() - test_start
    return results, total_time


def print_summary(results, total_time):
    successes = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in results]
    intents = [r["intent"] for r in results if r["intent"]]
    agentic_count = sum(1 for i in intents if i == "AgenticRAG")
    tokens_in = sum(r["input_tokens"] or 0 for r in results)
    tokens_out = sum(r["output_tokens"] or 0 for r in results)

    print("\n" + "=" * 55)
    print("RAG LOAD TEST REPORT (AgenticRAG-style queries)")
    print("=" * 55)
    print(f"  Requests:    {len(results)}")
    print(f"  Successful:  {len(successes)}")
    print(f"  Failed:      {len(results) - len(successes)}")
    print(f"  Total time:  {total_time:.2f}s")
    print(f"  Throughput:  {len(results) / total_time:.2f} req/s")
    if intents:
        print(f"  Routed to AgenticRAG: {agentic_count}/{len(intents)} "
              f"({100 * agentic_count / len(intents):.0f}%) — misroutes to KnowledgeRAG "
              f"mean the intent detector isn't recognizing these as multi-step queries.")
    if latencies:
        print(f"  Avg latency: {statistics.mean(latencies):.2f}s")
        print(f"  Median:      {statistics.median(latencies):.2f}s")
        print(f"  Min/Max:     {min(latencies):.2f}s / {max(latencies):.2f}s")
    print(f"  Tokens:      {tokens_in} in / {tokens_out} out")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Load test the /chat/ endpoint with AgenticRAG-style queries")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--session-id", required=True, help="Existing session_id with at least one indexed document")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("rag_results.json"))
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_file:
        queries = [l.strip() for l in args.queries_file.read_text().splitlines() if l.strip()]

    print(f"Target: {args.base_url}/chat/")
    print(f"Requests={args.requests} concurrency={args.concurrency}")

    results, total_time = asyncio.run(main_async(args, queries))
    print_summary(results, total_time)

    payload = {
        "endpoint": "rag", "base_url": args.base_url, "total_requests": args.requests,
        "concurrency": args.concurrency, "total_time": total_time, "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
