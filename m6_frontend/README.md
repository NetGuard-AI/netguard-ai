# Module M6 — NetGuard Application (Ronal)

Final reviewed M6 application package is supplied in the ChatGPT conversation as:
NetGuard_AI_M6_Ronal_Final_Integrated.zip

The application consumes the final M5/M7 REST contract and surfaces M3/M4
forecast/explainability plus M7 knowledge/help resources.

Final UI endpoints:
- GET /predict
- GET /rollout?k=8
- GET /security-zone
- GET /explain
- GET /alerts
- POST /alerts/trigger-attack
- GET /knowledge-center
- GET /help-resources

The UI does not load M1/M2 artifacts and does not call /traffic, /stats, or
/ws/live. No traffic or WebSocket values are fabricated.

Run from the supplied netguard-ai directory:
npm install
npm run typecheck
npm run build
npm run dev

The reviewed ZIP's SHA-256 is recorded in FINAL_DELIVERABLE_SHA256.txt.
