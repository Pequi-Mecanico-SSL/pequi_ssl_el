# A game event as carried in the referee message. Only the parts every event has;
# the full protobuf is available as JSON in `details`.
string type                          # GameEvent.Type name, e.g. BALL_LEFT_FIELD_TOUCH_LINE
string by_team                       # YELLOW, BLUE or UNKNOWN
string[] origin                      # who raised it (autoref name, GC, UI)
string details                       # JSON of the event payload (locations in metres)
