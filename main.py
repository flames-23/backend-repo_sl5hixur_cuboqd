import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from database import db, create_document, get_documents
import requests

app = FastAPI(title="Cold Outreach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Schemas
# -----------------------------
class Workspace(BaseModel):
    name: str
    owner_email: EmailStr

class Prospect(BaseModel):
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None

class SequenceStep(BaseModel):
    day_offset: int = Field(ge=0)
    subject: str
    body: str

class Campaign(BaseModel):
    name: str
    workspace_id: str
    sequence: List[SequenceStep]

class GenerateRequest(BaseModel):
    product: str
    audience: str
    tone: str = "friendly"
    call_to_action: str = "Book a quick call?"

class EmailPreview(BaseModel):
    subject: str
    body: str

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Cold Outreach API running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or ""
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response

@app.post("/workspaces")
def create_workspace(ws: Workspace):
    ws_id = create_document("workspace", ws.model_dump())
    return {"id": ws_id, **ws.model_dump()}

@app.post("/prospects")
def create_prospect(p: Prospect):
    pid = create_document("prospect", p.model_dump())
    return {"id": pid, **p.model_dump()}

@app.post("/campaigns")
def create_campaign(c: Campaign):
    cid = create_document("campaign", c.model_dump())
    return {"id": cid, **c.model_dump()}

@app.get("/campaigns")
def list_campaigns(limit: int = 20):
    items = get_documents("campaign", {}, limit)
    # Convert ObjectId to string
    for it in items:
        it["_id"] = str(it.get("_id"))
    return items

# -----------------------------
# AI Email Generation (OpenAI)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/generate", response_model=EmailPreview)
def generate_email(req: GenerateRequest):
    if not OPENAI_API_KEY:
        # Fallback simple template so UI works without a key
        subject = f"Quick idea about {req.product} for {req.audience}"
        body = (
            f"Hi there,\n\n"
            f"I noticed you're working with {req.audience}. We built {req.product} that could help a lot.\n\n"
            f"Would you be open to {req.call_to_action.lower()}\n\n"
            f"Best,\nYour Name"
        )
        return {"subject": subject, "body": body}

    try:
        # Use the new OpenAI Python SDK style via REST to avoid optional deps
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You write concise, high-converting cold emails."},
                {"role": "user", "content": (
                    "Generate a short cold email with subject and body as JSON. "
                    f"Product: {req.product}. Audience: {req.audience}. Tone: {req.tone}. "
                    f"CTA: {req.call_to_action}."
                )}
            ],
            "response_format": {"type": "json_object"}
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        # content should be a JSON string with subject/body
        import json
        try:
            parsed = json.loads(content)
            subject = parsed.get("subject") or "Quick question"
            body = parsed.get("body") or "Hi, quick note about our product."
        except Exception:
            subject = "Quick question"
            body = content
        return {"subject": subject, "body": body}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)[:150]}")

# -----------------------------
# Simple webhook to record events (analytics placeholder)
# -----------------------------
class Event(BaseModel):
    type: str
    properties: dict = {}

@app.post("/events")
def record_event(ev: Event):
    eid = create_document("event", ev.model_dump())
    return {"id": eid}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
