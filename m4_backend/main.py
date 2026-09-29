from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from . import db
from .api_schemas import AuthRequest, AuthResponse, IngestResponse, PredictResponse, ForecastResponse, SecurityZoneResponse, AlertResponse, EvidenceResponse
from .state_provider import PreparedStateProvider

from m3_forecast_xai.output.forecast_engine import rollout

SECRET = __import__("os").getenv("NETGUARD_JWT_SECRET", "change-this-demo-secret")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
DEFAULT_K = 5
HIGH_PROBABILITY = 0.70
app = FastAPI(title="NetGuard AI - M5 Risk, Alerts and Cyber Help")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

provider = PreparedStateProvider()
latest_forecast = None
last_device_id = None

KNOWLEDGE = [
    {"title":"Brute-force attacks","description":"Repeated authentication attempts may indicate credential guessing.","indicators":["Repeated connection attempts","Unusual authentication volume"]},
    {"title":"DDoS and DoS","description":"Large or abnormal traffic volumes can degrade availability.","indicators":["Traffic spikes","High packet rates","Many concurrent flows"]},
    {"title":"Network reconnaissance","description":"Scanning and probing can precede later intrusion activity.","indicators":["Repeated short connections","Unusual destination patterns"]},
]
HELP = [
    {"name":"CERT-In","purpose":"India national incident-response and cybersecurity guidance resource.","url":"https://www.cert-in.org.in/","phone":None},
    {"name":"National Cyber Crime Reporting Portal","purpose":"Official portal for reporting cybercrime in India.","url":"https://cybercrime.gov.in/","phone":"1930"},
]

@app.on_event("startup")
def startup():
    db.init_db()

def _decode_user(authorization):
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    try:
        payload = jwt.decode(authorization.split(" ",1)[1], SECRET, algorithms=["HS256"])
        return int(payload["sub"])
    except (JWTError, KeyError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def _optional_user(authorization: str | None = Header(default=None)):
    return _decode_user(authorization)

def _token(user_id):
    exp = datetime.now(timezone.utc) + timedelta(hours=8)
    return jwt.encode({"sub":str(user_id),"exp":exp}, SECRET, algorithm="HS256")

def _probability(result):
    values = result.get("attack_probability_timeline", [])
    return float(max(values)) if values else 0.0

def _zone(result):
    p = _probability(result)
    if p >= 0.70:
        return "red", p, "Forecasted attack probability is high."
    if p >= 0.30:
        return "yellow", p, "Forecast indicates elevated attack risk."
    return "green", p, "No elevated forecast risk is currently indicated."

def _confidence(result):
    p = _probability(result)
    if result.get("stage_confidence") == "confident" and p >= 0.70:
        return "high"
    if p >= 0.30:
        return "medium"
    if p > 0:
        return "low"
    return "unknown"

@app.get("/health")
def health():
    """Lightweight deployment health check that validates the model artifact."""
    try:
        if not provider.path.exists():
            raise RuntimeError(f"State bootstrap missing: {provider.path}")
        from m3_forecast_xai.output.forecast_engine import _load_model
        _load_model()
        return {"status": "ok", "model": "loaded", "state_bootstrap": str(provider.path)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/signup", response_model=AuthResponse)
def signup(payload: AuthRequest):
    if db.get_user(payload.email):
        raise HTTPException(status_code=409, detail="Account already exists")
    uid = db.create_user(payload.email, pwd.hash(payload.password))
    return {"access_token":_token(uid),"token_type":"bearer"}

@app.post("/login", response_model=AuthResponse)
def login(payload: AuthRequest):
    user = db.get_user(payload.email)
    if not user or not pwd.verify(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token":_token(user["id"]),"token_type":"bearer"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(record: dict, user_id: int | None = Depends(_optional_user)):
    global latest_forecast, last_device_id
    device_id = str(record.get("device_id","demo-device"))
    last_device_id = device_id
    db.upsert_device(device_id, user_id=user_id, name=record.get("device_name"))
    progress = provider.add_replay_flow()
    if not progress["state_completed"] or not provider.ready():
        return {"stored":True,"flow_count":progress["flow_count"],"state_count":progress["state_count"],"state_completed":progress["state_completed"],"forecast":None if not provider.ready() else latest_forecast}
    try:
        latest_forecast = rollout(provider.current_sequence(), DEFAULT_K)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if _probability(latest_forecast) >= HIGH_PROBABILITY:
        db.insert_alert(device_id,"high",f"Forecast risk is high; predicted stage: {latest_forecast['predicted_stage']}.","forecast")
    return {"stored":True,"flow_count":progress["flow_count"],"state_count":progress["state_count"],"state_completed":True,"forecast":latest_forecast}

@app.get("/predict", response_model=PredictResponse)
def predict():
    if latest_forecast is None:
        raise HTTPException(status_code=409, detail="No forecast is available yet")
    p=float(latest_forecast["attack_probability_timeline"][0]) if latest_forecast["attack_probability_timeline"] else 0.0
    return {"attack_detected":p>=0.50,"attack_probability":p,"confidence":_confidence(latest_forecast),"predicted_stage":latest_forecast["predicted_stage"],"trend":latest_forecast["trend"]}

@app.get("/rollout", response_model=ForecastResponse)
def rollout_endpoint(k: int = Query(default=5, ge=1, le=100)):
    global latest_forecast
    if not provider.ready():
        raise HTTPException(status_code=409, detail="Twenty prepared network states are required")
    try:
        latest_forecast=rollout(provider.current_sequence(),k)
    except (FileNotFoundError,ValueError,RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return latest_forecast

@app.get("/explain")
def explain_endpoint():
    if latest_forecast is None:
        raise HTTPException(status_code=409, detail="No forecast is available yet")
    return {"top_features":latest_forecast["top_features"],"explanation_text":latest_forecast["explanation_text"]}

@app.get("/security-zone", response_model=SecurityZoneResponse)
def security_zone():
    if latest_forecast is None:
        raise HTTPException(status_code=409, detail="No forecast is available yet")
    zone,p,reason=_zone(latest_forecast)
    return {"zone":zone,"attack_probability":p,"reason":reason}

@app.get("/alerts", response_model=list[AlertResponse])
def alerts(limit: int = Query(default=50, ge=1, le=500)):
    return db.list_alerts(limit)

@app.post("/alerts/trigger-attack")
def trigger_attack():
    global latest_forecast
    latest_forecast={"attack_probability_timeline":[0.85,0.90,0.94],"trend":"rising","predicted_stage":"Initial Access","stage_confidence":"confident","top_features":[{"feature":"SYN Flag Count","contribution":0.31},{"feature":"Flow Packets/s","contribution":0.24}],"explanation_text":"Demo attack mode is active; this does not alter or control the network."}
    aid=db.insert_alert(last_device_id,"high","Demo attack scenario triggered for testing the alert workflow.","demo")
    return {"triggered":True,"alert_id":aid,"forecast":latest_forecast}

@app.post("/incidents")
def create_incident():
    if latest_forecast is None:
        raise HTTPException(status_code=409, detail="No forecast is available yet")
    iid=db.create_incident(last_device_id,latest_forecast)
    return {"incident_id":iid,"status":"open"}

@app.get("/evidence-pack/{incident_id}", response_model=EvidenceResponse)
def evidence_pack(incident_id:int, format:str=Query(default="pdf", pattern="^(pdf|json)$")):
    incident=db.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    evidence=Path(__file__).resolve().parent/"evidence"
    evidence.mkdir(exist_ok=True)
    if format=="json":
        path=evidence/f"incident_{incident_id}.json"
        payload=dict(incident)
        payload["forecast"]=json.loads(payload.pop("forecast_json"))
        path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    else:
        path=evidence/f"incident_{incident_id}.pdf"
        styles=getSampleStyleSheet()
        doc=SimpleDocTemplate(str(path),pagesize=A4)
        story=[Paragraph("NetGuard AI - Incident Evidence Pack",styles["Title"]),Spacer(1,12),Paragraph(f"Incident ID: {incident_id}",styles["BodyText"]),Paragraph(f"Status: {incident['status']}",styles["BodyText"]),Spacer(1,12),Paragraph("Forecast evidence",styles["Heading2"]),Paragraph(json.dumps(json.loads(incident["forecast_json"]),indent=2).replace("\n","<br/>"),styles["Code"])]
        doc.build(story)
    return {"incident_id":incident_id,"download_format":format,"status":"generated","path":str(path)}

@app.get("/knowledge-center")
def knowledge_center():
    return KNOWLEDGE

@app.get("/help-resources")
def help_resources():
    return HELP

@app.get("/")
def root():
    return {"service":"NetGuard AI","module":"M5 Risk, Alerts and Security Intelligence + M7 Cyber Safety and Assistance","model_input":"prepared M1 network states","scaler_required":False,"placeholder_required":False,"docs":"/docs"}
