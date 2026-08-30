import json
import re
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

from biomni.agent import A1
from biomni.config import default_config
from bioset_biomni.prompt import DEFAULT_DATASET, build_prompt

app = Flask(__name__)

_agent: A1 | None = None
_agent_lock = threading.Lock()

# Max steps per mode for the label endpoint
_MODE_MAX_STEPS = {
    "minimal": 5,
    "db": 15,
    "full": 30,
}

# HGNC dataset config
_HGNC_GDRIVE_ID = "1znogniT4GLa_HieLXoE8mO42TA6-UAd_"
_HGNC_FILENAME = "hgnc_complete_set.tsv"
_HGNC_DESCRIPTION = (
    "HUGO Gene Nomenclature Committee (HGNC) complete gene set. "
    "Contains unique symbols and names for human loci, including protein coding genes, "
    "ncRNA genes and pseudogenes. Can give canonical names for biomarker names in CyCIF. "
    "You can use it if you encounter a gene you do not know about, to get its aliases and official names."
)
_DATA_DIR = Path(__file__).parent / "data"


def _ensure_hgnc(agent: A1) -> None:
    """Download the HGNC dataset if absent and register it with the agent."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / _HGNC_FILENAME

    if not dest.exists():
        print(f"[HGNC] Downloading to {dest} ...")
        try:
            import gdown
            gdown.download(id=_HGNC_GDRIVE_ID, output=str(dest), quiet=False)
        except Exception as e:
            print(f"[HGNC] Download failed: {e}")
            return

    agent.add_data({str(dest): _HGNC_DESCRIPTION})


def _parse_solution(text: str) -> dict:
    """Extract a JSON object from a <solution> block or bare text.

    Looks for <solution>...</solution> first, then finds the outermost
    { ... } within it and parses as JSON.
    """
    sol_match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    content = sol_match.group(1).strip() if sol_match else text.strip()

    # Find the outermost JSON object
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in solution. Raw output: {content[:300]}")

    return json.loads(json_match.group(0))


@app.post("/init")
def init():
    """Initialise the A1 agent.

    Body (JSON, all optional):
        llm     : model id for the main agent (default: claude-sonnet-4-6)
        db_llm  : model id for database queries (default: same as llm)
        mode    : "full" | "db" | "minimal"  (default: "full")
        dataset : dataset description used to specialise the prompt
                  (default: "melanoma CyCIF")
        api_key : override API key
    """
    data = request.get_json(force=True, silent=True) or {}

    llm = data.get("llm", "claude-sonnet-4-6")
    db_llm = data.get("db_llm") or llm
    mode = data.get("mode", "full")
    dataset = data.get("dataset", DEFAULT_DATASET)
    api_key = data.get("api_key") or None

    global _agent
    with _agent_lock:
        try:
            default_config.llm = db_llm
            kwargs = dict(llm=llm, mode=mode, custom_prompt=build_prompt(dataset))
            if api_key:
                kwargs["api_key"] = api_key
            _agent = A1(**kwargs)
            _ensure_hgnc(_agent)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


def _check_init():
    """Return an error response if the agent is not initialised, else None."""
    with _agent_lock:
        if _agent is None:
            return jsonify({"error": "Agent not initialised. Call POST /init first."}), 400
    return None


def _run(task_json: dict, image_b64: str | None, mode: str) -> tuple:
    """Build the prompt, run the agent, and return a (result_dict, None) or (None, error_response)."""
    max_steps = _MODE_MAX_STEPS.get(mode, _MODE_MAX_STEPS["full"])
    prompt = (
        "Return your answer ONLY as a JSON object inside a <solution> tag — "
        "no other text outside the tag.\n\n"
        f"{json.dumps(task_json, indent=2)}"
    )
    try:
        with _agent_lock:
            _, response = _agent.go(prompt, image=image_b64, max_steps=max_steps)
        return _parse_solution(response), None
    except ValueError as e:
        return None, (jsonify({"error": str(e)}), 422)
    except Exception as e:
        return None, (jsonify({"error": str(e)}), 500)


def _extract_common(data: dict) -> tuple[list | None, dict | None, str | None, str]:
    """Extract and validate fields shared by all task endpoints.

    Returns (markers, channel_stats, image_b64, mode).
    markers is None when missing so callers can return an error.
    """
    markers = data.get("markers") or None
    channel_stats = data.get("channel_stats") or None
    image_b64 = data.get("image") or None
    mode = data.get("mode", "full")
    return markers, channel_stats, image_b64, mode


@app.post("/upload")
def upload():
    """Upload a dataset file and register it with the agent.

    Multipart form fields:
        file        : the dataset file (required)
        description : plain-text description of the dataset (required)

    Returns: {"status": "ok", "filename": "<saved filename>"}
    """
    err = _check_init()
    if err:
        return err

    if "file" not in request.files:
        return jsonify({"error": "Missing required field: 'file'"}), 400
    description = request.form.get("description") or ""
    if not description:
        return jsonify({"error": "Missing required field: 'description'"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Uploaded file has no filename"}), 400

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / f.filename
    f.save(str(dest))

    with _agent_lock:
        _agent.add_data({str(dest): description})

    return jsonify({"status": "ok", "filename": f.filename})


@app.post("/label")
def label():
    """Generate biological labels for a set of markers.

    Body (JSON):
        markers       : list[str]  – e.g. ["SOX10:#FFFF00", "PRAME:#FF0000"]  (required)
        channel_stats : dict       – full channel statistics for the region      (optional)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                 (optional)

    Returns: {"labels": {...}, "overall": [...]}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400

    task_json = {"task": "label", "markers": markers}
    if channel_stats:
        task_json["channel_stats"] = channel_stats
    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/query")
def query():
    """Answer a free-form question about a set of markers.

    Body (JSON):
        markers       : list[str]  – e.g. ["SOX10:#FFFF00", "PRAME:#FF0000"]  (required)
        query         : str        – the question to answer                     (required)
        channel_stats : dict       – full channel statistics for the region     (optional)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                 (optional)

    Returns: {"answer": "..."}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)
    question = data.get("query") or None
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400
    if not question:
        return jsonify({"error": "Missing required field: 'query'"}), 400

    task_json = {"task": "query", "markers": markers, "query": question}
    if channel_stats:
        task_json["channel_stats"] = channel_stats
    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/suggest")
def suggest():
    """Recommend additional channels to enable alongside the currently selected markers.

    Body (JSON):
        markers       : list[str]  – currently selected markers                 (required)
        channel_stats : dict       – full channel statistics for the region     (optional)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                 (optional)

    Returns: {"suggestions": [{"channel": str, "reason": str, "priority": str}, ...]}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400

    task_json = {"task": "suggest", "markers": markers}
    if channel_stats:
        task_json["channel_stats"] = channel_stats
    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/plot")
def plot():
    """Explain the currently displayed UpSet or bar plot.

    Body (JSON):
        plot          : dict       – complete plot payload for current UI state (required)
        markers       : list[str]  – active markers with colors               (optional)
        channel_stats : dict       – full channel statistics for the region    (optional)
        query         : str        – optional user question about the plot      (optional)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                (optional)

    Returns: {"answer": "..."}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}

    plot_payload = data.get("plot")
    if not isinstance(plot_payload, dict) or not plot_payload:
        return jsonify({"error": "Missing required field: 'plot' (non-empty object)"}), 400

    mode = data.get("mode", "full")
    image_b64 = data.get("image") or None
    markers = data.get("markers") or []
    channel_stats = data.get("channel_stats")
    question = data.get("query") or None

    task_json = {
        "task": "plot",
        "plot": plot_payload,
        "markers": markers,
    }
    if channel_stats is not None:
        task_json["channel_stats"] = channel_stats
    if question:
        task_json["query"] = question

    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/bookmark")
def bookmark():
    """Suggest bookmark form text for the current view.

    Body (JSON):
        markers       : list[str]  – active markers with colors               (required)
        channel_stats : dict       – full channel statistics for the region    (optional)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                (optional)

    Returns: {"title": "...", "category": "...", "description": "..."}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)

    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400

    task_json = {
        "task": "bookmark",
        "markers": markers,
    }
    if channel_stats is not None:
        task_json["channel_stats"] = channel_stats

    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/explain")
def explain():
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}

    # Extracting data from request as before. We assume that all data is present in the request.
    markers, channel_stats, image_b64, _ = _extract_common(data)

    # Throw error as before
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400

    # We iterate through the modes manually
    modes = ["minimal", "db", "full"]

    # Leave-one-out: a baseline with everything enabled, then one row per
    # feature that removes only that feature.
    feature_sets = [
        (True, True, True),    # baseline
        (False, True, True),   # no markers
        (True, False, True),   # no stats
        (True, True, False),   # no image
    ]
    grid = [(mode, *feats) for mode in modes for feats in feature_sets]

    # Store the data in directory
    results_dir = Path(__file__).parent / "ablation"
    # Make sure the dir exists
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"explain_{int(time.time())}.jsonl"

    runs = []
    total = len(grid)

    # Iterates through all combinations
    for i, (mode, use_markers, use_stats, use_image) in enumerate(grid, 1):
        # Assemble the task json, now we set list to [] if use_markers is false
        task_json = {
            "task": "explain", 
            "markers": markers if use_markers else []
        }

        # If use_stats, include the channel stats. To make it simpler, we assume they are present
        if use_stats:
            task_json["channel_stats"] = channel_stats

        # If use_image, send the image. To make it simpler, we assume it is present
        img = image_b64 if use_image else None

        # Starts the timer
        start_time = time.time()

        # Run the query on Biomni
        result, err = _run(task_json, img, mode)

        # End timer, so we know how long the request took
        elapsed_time = time.time() - start_time

        # Extract answer
        answer = result.get("answer") if result else None

        record = {
            "run": i,
            "mode": mode,
            "markers": use_markers,
            "channel_stats": use_stats,
            "image": use_image,
            "answer": answer,
            "elapsed_s": elapsed_time,
        }
        runs.append(record)

        # Write result record to results_path
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Log all runs in the ablations
        if answer is not None:
            print(f'Run: {i}/{total}, mode={mode}, m={int(use_markers)}, s={int(use_stats)}, i={int(use_image)}, Time: {elapsed_time:.2f}s, Answer: {answer[:80]}')
        else:
            print(f'Answer is none in run: {i}/{total}')

    # Same return as before, crashes if we never iterate through the grid (no result is generated then).
    return err if err else jsonify(result)


def start_server(port: int = 5000, debug: bool = False):
    """Start the Biomni Flask server.

    Args:
        port  : TCP port to listen on (default 5000).
        debug : Enable Flask debug mode (default False).
    """
    app.run(host="0.0.0.0", port=port, debug=debug)
