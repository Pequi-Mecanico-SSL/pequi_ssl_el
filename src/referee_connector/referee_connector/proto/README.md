# Game-controller protocol files

The `.proto` files in this folder are copied from the ssl-game-controller project
(https://github.com/RoboCup-SSL/ssl-game-controller, folder `proto/`), licensed under the
GNU General Public License v3.0 (GPL-3.0). They are unmodified apart from their `import` paths,
which are flattened to this folder (`import "state/ssl_gc_common.proto"` became
`import "ssl_gc_common.proto"`).

The `*_pb2.py` files are generated from them with protoc:

    protoc --python_out=. ssl_gc_common.proto ssl_gc_geometry.proto ssl_gc_game_event.proto ssl_gc_referee_message.proto
