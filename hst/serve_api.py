"""HTTP middleman for the HST classification pipeline.

Thin transport layer: receive an image, hand it to one of the retrieval
pipelines' classify() functions, return that pipeline's standard result dict as
JSON. It holds NO model logic — VLM endpoint, embeddings, corpus, rerank all
live behind classify() in the test_infer* scripts. To deploy against different
models later, change those scripts; this API is untouched as long as
classify(image_path) -> dict keeps its shape.

Endpoints:
  GET  /health           -> {"status": "ok", "flows": [...]}
  POST /classify         -> run a pipeline on an uploaded image
       multipart form:
         image       (file, required)   the product photo
         flow        (str,  optional)   "normal" (default) | "dense" | "ensemble"
         target_hs6  (str,  optional)   NULLABLE HS6 to verify against (verify
                                        pipeline TBD; today normalized + echoed)

Response JSON (uniform across flows):
  {flow, hs6, description, query, caption, chapters, candidates[], meta,
   verification, error?}
  meta.target_hs6 echoes the normalized target; verification is null until the
  verify pipeline lands.

Run (vision conda env, needs the VLM served on the endpoint the scripts target):
  uvicorn hst.serve_api:app --host 0.0.0.0 --port 9000
from the repo root, or:
  ~/miniforge3/envs/vision/bin/python -m uvicorn hst.serve_api:app \
      --host 0.0.0.0 --port 9000
"""
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Import the three pipelines. Each exposes classify(image, verbose=False)->dict.
# They live in the repo root (one dir up from hst/); make sure it's importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test_infer            # noqa: E402  "normal": flat KNN + reinfer + rerank
import test_infer_dense      # noqa: E402  "dense": chapter-gated KNN + rerank
import test_infer_ensemble   # noqa: E402  "ensemble": two-VLM RRF fusion

# flow name -> the module whose classify() runs it. Add new flows here only.
FLOWS = {
    "normal": test_infer,
    "dense": test_infer_dense,
    "ensemble": test_infer_ensemble,
}
DEFAULT_FLOW = "normal"

app = FastAPI(title="HST Classifier API", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok", "flows": list(FLOWS), "default_flow": DEFAULT_FLOW}


@app.post("/classify")
async def classify(
    image: UploadFile = File(...),
    flow: str = Form(DEFAULT_FLOW),
    target_hs6: str = Form(None),
):
    """Run a retrieval pipeline on an uploaded image.

    Form fields:
      image       (file, required)   the product photo
      flow        (str,  optional)   "normal" (default) | "dense" | "ensemble"
      target_hs6  (str,  optional)   an HS6 the caller wants VERIFIED against the
                                     image. NULLABLE — the app's generation tab
                                     omits it; the verification tab sends it. The
                                     verification pipeline is not built yet, so
                                     for now the code is only normalized and
                                     echoed back (`meta.target_hs6`) and the
                                     `verification` block is returned as null.
                                     A future verify-only flow reads this field.
    """
    flow = (flow or DEFAULT_FLOW).strip().lower()
    module = FLOWS.get(flow)
    if module is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown flow {flow!r}; valid: {list(FLOWS)}",
        )

    # Normalize the optional target code to NNNN.NN so the app + future verify
    # pipeline get a canonical form. Blank/None -> None (generation tab).
    target = (target_hs6 or "").strip()
    target = test_infer.norm6(target) if target else None

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image upload")

    # classify() takes a filesystem path (it base64s the bytes itself). Persist
    # the upload to a temp file, run, then clean up.
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        result = module.classify(tmp.name, verbose=False)
    except Exception as e:  # pipeline blew up (VLM down, etc) — surface as 502
        raise HTTPException(status_code=502, detail=f"pipeline error: {e}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    # Forward-compat verification seam. Today: echo the normalized target and
    # return verification=null (no verify pipeline yet). Later: a verify flow
    # fills this block with {verdict, rank, confidence, ...} — the app already
    # reads result["verification"], so it needs no change when that lands.
    if isinstance(result, dict):
        result.setdefault("meta", {})
        if isinstance(result["meta"], dict):
            result["meta"]["target_hs6"] = target
        result["verification"] = None

    return JSONResponse(result)
