# Layout critic

Image 1 is the background without editable content. Image 2 is the current slide with region IDs.
Return up to three editable regions that materially hide a key visual subject or another editable unit, worst first.
Ignore intentional holder interiors and minor decorative overlap.

Return only: `{"move_region_ids":["region_id"]}`. Use an empty list when no move is needed.
