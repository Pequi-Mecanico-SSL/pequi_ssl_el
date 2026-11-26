# Published by the router: which behavior currently drives /<team>/robot<i>/cmd_vel.
std_msgs/Header header
string behavior                      # e.g. rl, stop, kickoff, free_kick, halt
uint32 activation_id                 # increments on every switch; behaviors reset when it changes
bool ours                            # the kick / placement in progress is for our team
uint8 for_team                       # 0 UNKNOWN, 1 YELLOW, 2 BLUE
uint8 game_state                     # sslrules.GameState value that selected this behavior
uint8[] benched                     # robot indices relayed from `bench` instead (cards / too many robots)
