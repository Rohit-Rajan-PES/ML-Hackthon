import logging, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipeline import load_pipeline, list_trip_ids, predict_trip
from scoring import score_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("pathsnap")

state = {}


@asynccontextmanager
async def lifespan(app):
    state["pipe"] = load_pipeline()
    state["trips"] = list_trip_ids()
    log.info("Model + HMM config loaded. %d evaluator trips: %s", len(state["trips"]), state["trips"])
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipe" in state, "n_trips": len(state.get("trips", []))}


@app.get("/trips")
def trips():
    return {"trips": state["trips"]}


@app.post("/evaluate")
def evaluate(req: dict):
    trip_id = req.get("trip_id")
    if not trip_id:
        raise HTTPException(status_code=422, detail="trip_id missing")
    if trip_id not in state["trips"]:
        raise HTTPException(status_code=404, detail="Unknown trip_id")
    log.info("REQUEST received for trip %s", trip_id)
    t0 = time.time()
    try:
        pred = predict_trip(state["pipe"], trip_id)          # no labels used
        t_pred = time.time() - t0
        result = score_trip(trip_id, pred)                   # server-side only, after prediction
    except Exception as e:
        log.exception("FAILED trip=%s", trip_id)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    m = result["metrics"]
    log.info("STAGES trip=%s %s", trip_id, pred["timings"])
    log.info("DONE trip=%s classifier_acc=%s hmm_acc=%s continuity=%s predict=%.2fs total=%.2fs",
             trip_id, m["classifier_top1"], m["hmm_top1"], m["route_continuity"],
             t_pred, time.time() - t0)
    return {"trip_id": trip_id, **pred, **result}


@app.get("/evaluate")
def evaluate_query(trip_id: str):
    return evaluate({"trip_id": trip_id})
