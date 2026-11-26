# The SSL referee message (224.5.23.1:10003), decoded, SI units.
# Enum values are the wire integers; names are the protobuf enum names.
std_msgs/Header header
uint64 packet_timestamp              # microseconds, GC clock
string source_identifier
string match_type
uint8 stage
string stage_name
float32 stage_time_left              # seconds (negative when over)
uint8 command
string command_name
uint32 command_counter
uint64 command_timestamp             # microseconds, GC clock
bool has_next_command
uint8 next_command
string next_command_name
bool has_designated_position
float32 designated_x                 # metres (wire is mm)
float32 designated_y
bool blue_team_on_positive_half
float32 current_action_time_remaining   # seconds; negative once the action's timer ran out
pequi_msgs/TeamInfo yellow
pequi_msgs/TeamInfo blue
pequi_msgs/GameEvent[] game_events
