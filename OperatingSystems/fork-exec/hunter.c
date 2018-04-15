#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/poll.h>
#include <sys/socket.h>

#define PIPE(fd) socketpair(AF_UNIX, SOCK_STREAM, PF_UNIX, fd)


typedef struct coordinate
{
    int x;
    int y;
} coordinate;

typedef struct server_message
{
    coordinate pos;
    coordinate adv_pos;
    int object_count;
    coordinate object_pos[4];
} server_message;

typedef struct ph_message
{
    coordinate move_request;
} ph_message;

int MD(int x0, int y0, int x1, int y1)
{
    return abs(x1-x0) + abs(y1-y0);
}

int isCollision(int x, int y, int object_count, coordinate objects[4])
{
    int i;
    for(i=0; i<object_count; ++i){
        if(x == objects[i].x && y == objects[i].y)
            return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int readBytes, currentMD;
    coordinate me, requestMe;
    ph_message outgoing;
    server_message incoming;
    struct pollfd pfds[] = {
            {STDIN_FILENO, POLLIN},
            {STDOUT_FILENO, POLLOUT}
        };
    int width, height;
    sscanf(argv[1], "%i", &width);
    sscanf(argv[2], "%i", &height);
    
    while(1){
        poll(pfds, 2, -1);
        
        if(pfds[0].revents & POLLIN){
            readBytes = read(STDIN_FILENO, &incoming, sizeof(incoming));
            if(readBytes == sizeof(incoming)){
                me = incoming.pos;
                requestMe = me;
                currentMD = MD(me.x, me.y, incoming.adv_pos.x, incoming.adv_pos.y);
                if(me.x-1 >= 0)
                    if(!isCollision(me.x-1, me.y, incoming.object_count, incoming.object_pos))
                        if(MD(me.x-1, me.y, incoming.adv_pos.x, incoming.adv_pos.y) < currentMD){
                            requestMe.x = me.x-1;
                            requestMe.y = me.y;
                        }
                if(me.x+1 < width)
                    if(!isCollision(me.x+1, me.y, incoming.object_count, incoming.object_pos))
                        if(MD(me.x+1, me.y, incoming.adv_pos.x, incoming.adv_pos.y) < currentMD){
                            requestMe.x = me.x+1;
                            requestMe.y = me.y;
                        }
                if(me.y-1 >= 0)
                    if(!isCollision(me.x, me.y-1, incoming.object_count, incoming.object_pos))
                        if(MD(me.x, me.y-1, incoming.adv_pos.x, incoming.adv_pos.y) < currentMD){
                            requestMe.x = me.x;
                            requestMe.y = me.y-1;
                        }
                if(me.y+1 < height)
                    if(!isCollision(me.x, me.y+1, incoming.object_count, incoming.object_pos))
                        if(MD(me.x, me.y+1, incoming.adv_pos.x, incoming.adv_pos.y) < currentMD){
                            requestMe.x = me.x;
                            requestMe.y = me.y+1;
                        }
                outgoing.move_request = requestMe;
                if(pfds[1].revents & POLLOUT){
                    while(write(STDOUT_FILENO, &outgoing, sizeof(outgoing)) != sizeof(outgoing))
                        fflush(0);
                }
            }
        }
        usleep(10000*(1+rand()%9));
    }
    
    return 1;
}