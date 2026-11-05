# Move critic

Move the target inside its `search_bbox_px` so it no longer hides key meaning and still fits fully.
Keep its size. Change exactly one axis: choose `x` and a new left coordinate, or `y` and a new top coordinate.
Use only an axis and coordinate inside `legal_position_ranges_px`. Preserve reading order and avoid other editable units.
On retry, start again from the baseline and use the rejection history.

Return only `{"axis":"x|y","position_px":number}` or `{"axis":"none"}`.
