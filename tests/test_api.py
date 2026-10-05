import json
import threading
import time

from openai import OpenAI


def test_tags_and_models(client):
    names = [m["name"] for m in client.get("/api/tags").json()["models"]]
    assert names == ["alpha", "beta"]
    assert client.get("/api/tags").json()["models"][0]["size"] == 100
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["alpha", "beta"]


def test_ps_empty_then_loaded(client):
    assert client.get("/api/ps").json()["models"] == []
    client.post("/api/generate", json={"model": "alpha", "prompt": "hi", "stream": False})
    ps = client.get("/api/ps").json()
    assert ps["models"][0]["name"] == "alpha" and ps["models"][0]["pid"] > 0
    assert ps["queue"]["queue_depth"] == 0


def test_ollama_chat_stream_and_nonstream(client):
    body = {"model": "alpha:latest", "messages": [{"role": "user", "content": "hello world"}]}
    r = client.post("/api/chat", json=body)
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert "".join(l["message"]["content"] for l in lines) == "alpha: hello world "
    assert lines[-1]["done"] and lines[-1]["done_reason"] == "stop"
    r = client.post("/api/chat", json={**body, "stream": False}).json()
    assert r["message"]["content"] == "alpha: hello world" and r["done"]


def test_generate_and_raw(client):
    r = client.post("/api/generate", json={"model": "alpha", "prompt": "x y", "stream": False}).json()
    assert r["response"] == "alpha: x y"
    r = client.post("/api/generate", json={"model": "alpha", "prompt": "x", "raw": True, "stream": False}).json()
    assert r["response"] == "alpha: x"


def test_errors(client):
    r = client.post("/api/chat", json={"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404 and "not found" in r.json()["error"]
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found"
    r = client.post("/api/generate", json={"model": "alpha", "prompt": "boom", "stream": True})
    assert r.status_code == 500 and r.json()["error"] == "engine exploded"
    assert client.get("/api/ps").json()["queue"]["queue_depth"] == 0


def test_swap_and_stop(client):
    client.post("/api/generate", json={"model": "alpha", "prompt": "x", "stream": False})
    pid_a = client.get("/api/ps").json()["models"][0]["pid"]
    client.post("/api/generate", json={"model": "beta", "prompt": "x", "stream": False})
    ps = client.get("/api/ps").json()
    assert ps["models"][0]["name"] == "beta" and ps["models"][0]["pid"] != pid_a
    assert client.post("/api/stop").json() == {"stopped": True}
    assert client.get("/api/ps").json()["models"] == []
    assert client.post("/api/stop").json() == {"stopped": False}


def test_concurrent_mixed_models_all_complete(client):
    results = {}

    def go(i, model):
        r = client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": str(i)}]})
        results[i] = r.json()["choices"][0]["message"]["content"]

    threads = [threading.Thread(target=go, args=(i, m)) for i, m in enumerate(["alpha", "beta", "alpha", "beta"])]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join(60)
    assert results == {0: "alpha: 0", 1: "beta: 1", 2: "alpha: 2", 3: "beta: 3"}


def test_openai_sdk(client):
    # Drive the SDK through the TestClient's transport.
    sdk = OpenAI(base_url="http://testserver/v1", api_key="x", http_client=client)
    out = sdk.chat.completions.create(model="alpha", messages=[{"role": "user", "content": "hey"}])
    assert out.choices[0].message.content == "alpha: hey"
    chunks = sdk.chat.completions.create(model="alpha", messages=[{"role": "user", "content": "a b"}], stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "alpha: a b "
    comp = sdk.completions.create(model="alpha", prompt="p")
    assert comp.choices[0].text == "alpha: p"


def test_curl_style_form_content_type_and_bad_json(client):
    r = client.post("/api/generate", content='{"model":"alpha","prompt":"hi","stream":false}',
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 200 and r.json()["response"] == "alpha: hi"
    r = client.post("/api/generate", content="nope")
    assert r.status_code == 400 and "invalid JSON" in r.json()["error"]
    r = client.post("/v1/completions", content="[1]")
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
