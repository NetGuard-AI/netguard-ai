# M6 integration review

Reviewed the supplied Ronal ZIP against the final M5/M7 backend and the
provided M2/M3/M4 contracts.

Fixed integration flags:
1. Corrected M5 dashboard naming to M6 application.
2. Replaced the unavailable live-traffic placeholder with actual M4
   explainability data from /explain.
3. Added M7 Knowledge Center and Cyber Safety & Assistance using
   /knowledge-center and /help-resources.
4. Reduced forecast polling from 3 seconds to 10 seconds.
5. Kept M2 context facts separate from classification metrics.
6. Removed frontend assumptions about /traffic, /stats and /ws/live.

The final ZIP is delivered in the ChatGPT conversation. The connected
GitHub write interface accepts UTF-8 repository files but does not provide a
local-binary ZIP upload operation, so the SHA-256 is recorded here for
verification.
