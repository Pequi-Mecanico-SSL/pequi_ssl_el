Build with:
```
docker compose build bash simulator control
```
And run simulator with:
```
docker compose run simulator
```
To build for the raspberry pi 4, you can build all services:
```
docker compose build bash simulator control ping pong robot
```

To test communication latency, run `docker compose run ping` on the computer and `docker compose run pong` on the raspberry pi.
