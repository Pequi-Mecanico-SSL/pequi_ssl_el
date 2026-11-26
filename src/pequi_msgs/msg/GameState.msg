# Derived game state (what the rules care about), from the referee stream plus the ball position.
# state / next_state values are sslrules.GameState: 0 UNKNOWN, 1 HALT, 2 STOP, 3 RUNNING, 4 FREE_KICK,
# 5 KICKOFF, 6 PENALTY, 7 TIMEOUT, 8 BALL_PLACEMENT, 9 BREAK, 10 POST_GAME.
# team values: 0 UNKNOWN, 1 YELLOW, 2 BLUE.
std_msgs/Header header
uint8 state
string state_name
uint8 for_team                       # whose kick / placement / timeout it is
string for_team_name
bool has_next_state
uint8 next_state
uint8 next_for_team
bool ball_in_play
float32 ball_in_play_since           # node clock seconds; 0 if not in play
uint8 command                        # wire command that produced this state
uint32 command_counter
uint8 stage
bool has_placement_position
float32 placement_x                  # metres
float32 placement_y
bool blue_team_on_positive_half
