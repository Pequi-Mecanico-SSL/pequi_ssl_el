# One team's block of the SSL referee message (Referee.TeamInfo).
string name
uint32 score
uint32 red_cards
uint32 yellow_cards
float32[] yellow_card_times          # seconds remaining per active yellow card
uint32 timeouts                      # timeouts left
float32 timeout_time                 # seconds of timeout budget left
uint32 goalkeeper                    # vision id of the keeper
uint32 foul_counter
uint32 ball_placement_failures
bool can_place_ball
uint32 max_allowed_bots
bool bot_substitution_intent
bool ball_placement_failures_reached
