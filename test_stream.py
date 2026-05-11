import httpx
import json

def test_streaming_endpoint():
    print("Testing SSE streaming at /api/chat/stream...\n")
    url = "http://localhost:8000/api/chat/stream"
    payload = {
        "question": "كيف أتحصل على رخصة تجارية؟",
        "rag_type": "agentic"
    }

    try:
        with httpx.stream("POST", url, json=payload, timeout=300) as response:
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.read().decode()}")
                return
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        if event["type"] == "status":
                            print(f"[STATUS] {event['message']}")
                        elif event["type"] == "token":
                            print(f"{event['content']}", end="", flush=True)
                        elif event["type"] == "done":
                            print("\n\n[DONE] Sources:", event.get("sources", []))
                        elif event["type"] == "error":
                            print(f"\n[ERROR] {event['message']}")
                    except json.JSONDecodeError:
                        print(f"Raw data: {data_str}")
                elif line.strip():
                    print(f"Skipped line: {line}")
    except httpx.ConnectError:
        print("Could not connect to server. Ensure uvicorn is running on port 8000.")

if __name__ == "__main__":
    test_streaming_endpoint()
