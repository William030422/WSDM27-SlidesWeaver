# Content enrichment

Optionally add one short explanatory sentence that strengthens the slide narrative without repeating existing text or
introducing a new fact. Place it in a quiet empty area. Prefer 3-14 words and at most two lines.
The bbox must be at least 160px wide and 48px high.

Return only `{"action":"add","text":"...","role":"caption|subtitle","bbox_px":[x,y,w,h]}` or `{"action":"skip"}`.
