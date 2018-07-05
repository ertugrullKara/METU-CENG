#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>

#define PIPE(fd) socketpair(AF_UNIX, SOCK_STREAM, PF_UNIX, fd)
#define PARENT_END 0
#define CHILD_END 1
#define X_COORD_IDX 0
#define Y_COORD_IDX 1
#define ENERGY_IDX 2
#define PID_IDX 3

char *hunterFilename = "./hunter";
char *preyFilename = "./prey";
char **map;
int **hunters;
int **preys;
int **obstacles;
int **pipefds;
int width, height, num_obstacles, num_hunters, num_preys, energy;
int hunter_remaining, prey_remaining;

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

void _free2dArrayChar(int dim, char **twodimarr)
{
    int i;
    for(i=0; i<dim; i++) {
        free(twodimarr[i]);
    }
    free(twodimarr);
}

void _free2dArrayInt(int dim, int **twodimarr)
{
    int i;
    for(i=0; i<dim; i++) {
        free(twodimarr[i]);
    }
    free(twodimarr);
}

void _beforeExit()
{
    // return;
    int num_fds = (num_hunters+num_preys);
    int i;
    _free2dArrayChar(width, map);
    _free2dArrayInt(num_hunters, hunters);
    _free2dArrayInt(num_preys, preys);
    _free2dArrayInt(num_fds, pipefds);
    kill(0, SIGKILL);
    waitpid(-1, NULL, WNOHANG);
    for(i=0; i<num_fds; ++i){
        close(pipefds[i][CHILD_END]);
        close(pipefds[i][PARENT_END]);
    }
}

void printMap()
{
    int i, j;
    
    printf("+");
    for(i=0; i<width; ++i)
    {
        printf("-");
    }
    printf("+\n");
    
    for(i=0; i<width; ++i)
    {
        printf("|");
        for(j=0; j<height; ++j)
        {
            if(map[i][j])
                printf("%c", map[i][j]);
            else
                printf(" ");
        }
        printf("|\n");
    }
    
    printf("+");
    for(i=0; i<width; ++i)
    {
        printf("-");
    }
    printf("+\n");
    fflush(0);
}

int MD(int x0, int y0, int x1, int y1)
{
    return abs(x1-x0) + abs(y1-y0);
}

coordinate getClosestPrey(int hunter_x, int hunter_y)
{
    int i;
    coordinate closestAdv;
    int prevMD=INT_MAX;
    int md;
    
    for(i=0; i<num_preys; ++i){
        if(preys[i][ENERGY_IDX] == -1)
            continue;
        md = MD(hunter_x, hunter_y, preys[i][X_COORD_IDX], preys[i][Y_COORD_IDX]);
        if(md < prevMD){
            prevMD = md;
            closestAdv.x = preys[i][X_COORD_IDX];
            closestAdv.y = preys[i][Y_COORD_IDX];
        }
    }
    return closestAdv;
}

coordinate getClosestHunter(int prey_x, int prey_y)
{
    int i;
    coordinate closestAdv;
    int prevMD=INT_MAX;
    int md;
    
    for(i=0; i<num_hunters; ++i){
        if(hunters[i][ENERGY_IDX] == -1)
            continue;
        md = MD(prey_x, prey_y, hunters[i][X_COORD_IDX], hunters[i][Y_COORD_IDX]);
        if(md < prevMD){
            prevMD = md;
            closestAdv.x = hunters[i][X_COORD_IDX];
            closestAdv.y = hunters[i][Y_COORD_IDX];
        }
    }
    return closestAdv;
}

int isCollision(int x, int y, char senderType)
{
    if(map[x][y] == 'X' || map[x][y] == senderType){
        return 1;
    }
    return 0;
}

int findPrey(int x, int y)
{
    int i;
    for(i=0; i<num_preys; ++i)
        if(preys[i][X_COORD_IDX] == x && preys[i][Y_COORD_IDX] == y && preys[i][ENERGY_IDX] != -1)
            return i;
    return -1;
}

int findHunter(int x, int y)
{
    int i;
    for(i=0; i<num_hunters; ++i)
        if(hunters[i][X_COORD_IDX] == x && hunters[i][Y_COORD_IDX] == y && hunters[i][ENERGY_IDX] != -1)
            return i;
    return -1;
}

int getPreyEnergyFrom(int x, int y)
{
    int tmp;
    int prey = findPrey(x, y);
    if(prey == -1)
        return 0;
    tmp = preys[prey][ENERGY_IDX];
    return tmp;
}

void setNeighbouringObjects(int agent_x, int agent_y, server_message* outgoing,
                            char senderType)
{
    int neighbourIdx = 0;
    int obj_cnt = 0;
    if(agent_x+1 < width)
        if(map[agent_x+1][agent_y] == 'X' || map[agent_x+1][agent_y] == senderType){
            ++obj_cnt;
            outgoing->object_pos[neighbourIdx].x = agent_x+1;
            outgoing->object_pos[neighbourIdx++].y = agent_y;
        }
    if(agent_x-1 >= 0)
        if(map[agent_x-1][agent_y] == 'X' || map[agent_x-1][agent_y] == senderType){
            ++obj_cnt;
            outgoing->object_pos[neighbourIdx].x = agent_x-1;
            outgoing->object_pos[neighbourIdx++].y = agent_y;
        }
    if(agent_y+1 < height)
        if(map[agent_x][agent_y+1] == 'X' || map[agent_x][agent_y+1] == senderType){
            ++obj_cnt;
            outgoing->object_pos[neighbourIdx].x = agent_x;
            outgoing->object_pos[neighbourIdx++].y = agent_y+1;
        }
    if(agent_y-1 >= 0)
        if(map[agent_x][agent_y-1] == 'X' || map[agent_x][agent_y-1] == senderType){
            ++obj_cnt;
            outgoing->object_pos[neighbourIdx].x = agent_x;
            outgoing->object_pos[neighbourIdx++].y = agent_y-1;
        }
    outgoing->object_count = obj_cnt;
}

server_message getOutgoingMessageHunter(int hunter_idx)
{
    server_message outgoing;
    outgoing.pos.x = hunters[hunter_idx][X_COORD_IDX];
    outgoing.pos.y = hunters[hunter_idx][Y_COORD_IDX];
    outgoing.adv_pos = getClosestPrey(outgoing.pos.x, outgoing.pos.y);
    setNeighbouringObjects(outgoing.pos.x, outgoing.pos.y, &outgoing, 'H');
    return outgoing;
}

server_message getOutgoingMessagePrey(int prey_idx)
{
    server_message outgoing;
    outgoing.pos.x = preys[prey_idx][X_COORD_IDX];
    outgoing.pos.y = preys[prey_idx][Y_COORD_IDX];
    outgoing.adv_pos = getClosestHunter(outgoing.pos.x, outgoing.pos.y);
    setNeighbouringObjects(outgoing.pos.x, outgoing.pos.y, &outgoing, 'P');
    return outgoing;
}


int main(void)
{
    int i, x, y;
    int tmp;
    size_t readBytes;
    pid_t pid;
    struct pollfd *pfds;
    
    scanf("%d %d", &width, &height);
    map = malloc(width*sizeof(char*));
    for(i=0; i<width; i++)
        map[i] = malloc(height*sizeof(char));
    
    scanf("%d", &num_obstacles);
    obstacles = malloc(num_obstacles*sizeof(int*));
    for(i=0; i<num_obstacles; ++i) {
        obstacles[i] = malloc(2*sizeof(int));
        scanf("%d %d", &x, &y);
        obstacles[i][X_COORD_IDX] = x;
        obstacles[i][Y_COORD_IDX] = y;
        map[x][y] = 'X';
    }
    
    scanf("%d", &num_hunters);
    hunter_remaining = num_hunters;
    hunters = malloc(num_hunters*sizeof(int*));
    for(i=0; i<num_hunters; ++i) {
        hunters[i] = malloc(4*sizeof(int));
        scanf("%d %d %d", &x, &y, &energy);
        hunters[i][X_COORD_IDX] = x;
        hunters[i][Y_COORD_IDX] = y;
        hunters[i][ENERGY_IDX] = energy;
        map[x][y] = 'H';
    }
    
    scanf("%d", &num_preys);
    prey_remaining = num_preys;
    preys = malloc(num_preys*sizeof(int*));
    for(i=0; i<num_preys; ++i) {
        preys[i] = malloc(4*sizeof(int));
        scanf("%d %d %d", &x, &y, &energy);
        preys[i][X_COORD_IDX] = x;
        preys[i][Y_COORD_IDX] = y;
        preys[i][ENERGY_IDX] = energy;
        map[x][y] = 'P';
    }
    
    pipefds = malloc((num_hunters+num_preys)*sizeof(int*));
    pfds = malloc(sizeof(struct pollfd)*(num_hunters+num_preys)*2);
    int pfd_idx=0;
    for(i=0; i<(num_hunters+num_preys); ++i) {
        pipefds[i] = malloc(2*sizeof(int));
        PIPE(pipefds[i]);
        pfds[pfd_idx].fd = pipefds[i][PARENT_END];
        pfds[pfd_idx++].events = POLLIN;
        pfds[pfd_idx].fd = pipefds[i][PARENT_END];
        pfds[pfd_idx++].events = POLLOUT;
    }
    
    char argWidth[10], argHeight[10];
    char *argv[4];
    snprintf(argWidth, sizeof(argWidth), "%d", width);
    snprintf(argHeight, sizeof(argHeight), "%d", height);
    argv[0] = hunterFilename;
    argv[1] = argWidth;
    argv[2] = argHeight;
    argv[3] = NULL;
    for(i=0; i<num_hunters; ++i) {
        pid = fork();
        if(pid) {
            //parent
            hunters[i][PID_IDX] = pid;
            close(pipefds[i][CHILD_END]);
        }
        else {
            close(pipefds[i][PARENT_END]);
            dup2(pipefds[i][CHILD_END], STDIN_FILENO);
            dup2(pipefds[i][CHILD_END], STDOUT_FILENO);
            close(pipefds[i][CHILD_END]);
            
            execvp(argv[0], argv);
            _exit(EXIT_FAILURE);
        }
    }
    argv[0] = preyFilename;
    for(i=0; i<num_preys; ++i) {
        pid = fork();
        if(pid) {
            //parent
            preys[i][PID_IDX] = pid;
            close(pipefds[num_hunters+i][CHILD_END]);
        }
        else {
            close(pipefds[num_hunters+i][PARENT_END]);
            dup2(pipefds[num_hunters+i][CHILD_END], STDIN_FILENO);
            dup2(pipefds[num_hunters+i][CHILD_END], STDOUT_FILENO);
            close(pipefds[num_hunters+i][CHILD_END]);
            
            execvp(argv[0], argv);
            _exit(EXIT_FAILURE);
        }
    }
    
    ph_message incoming;
    server_message outgoing;
    int mapUpdated = 1;
    int agent_idx, fd_idx;
    
    for(agent_idx=0; agent_idx<num_hunters; ++agent_idx){
        outgoing = getOutgoingMessageHunter(agent_idx);
        while(write(pipefds[agent_idx][PARENT_END], &outgoing, sizeof(outgoing)) != sizeof(outgoing))
            ;
    }
    for(agent_idx=0; agent_idx<num_preys; ++agent_idx){
        outgoing = getOutgoingMessagePrey(agent_idx);
        while(write(pipefds[num_hunters+agent_idx][PARENT_END], &outgoing, sizeof(outgoing)) != sizeof(outgoing))
            ;
    }
    
    int pfd_cnt = (num_hunters+num_preys)*2;
    while(hunter_remaining > 0 && prey_remaining > 0)
    {
        i = poll(pfds, pfd_cnt, 0);
        if(i < 0) printf("ERROR");
        if(mapUpdated){
            printMap(width, height, map);
            mapUpdated = 0;
        }
        for(agent_idx=0, fd_idx=0; agent_idx<num_hunters; ++agent_idx, fd_idx+=2){
            if(hunters[agent_idx][ENERGY_IDX] == -1)
                continue;
            if(pfds[fd_idx].revents & POLLIN){
                pfds[fd_idx].revents = 0;
                readBytes = read(pipefds[agent_idx][PARENT_END], &incoming, sizeof(incoming));
                // printf("Hunter %d\n", hunters[agent_idx][ENERGY_IDX]);
                if(readBytes == sizeof(incoming)) {
                    if(isCollision(incoming.move_request.x, incoming.move_request.y, 'H'))
                        ;
                    else{
                        map[hunters[agent_idx][X_COORD_IDX]][hunters[agent_idx][Y_COORD_IDX]] = ' ';
                        mapUpdated = 1;
                        --hunters[agent_idx][ENERGY_IDX];
                        tmp = getPreyEnergyFrom(incoming.move_request.x, incoming.move_request.y);
                        if(tmp){
                            hunters[agent_idx][ENERGY_IDX] += tmp;
                            --prey_remaining;
                            int eatenIdx = findPrey(incoming.move_request.x, incoming.move_request.y);
                            preys[eatenIdx][ENERGY_IDX] = -1;
                            
                            kill(preys[eatenIdx][PID_IDX], SIGTERM);
                            waitpid(preys[eatenIdx][PID_IDX], NULL, WNOHANG);
                            close(pipefds[num_hunters+eatenIdx][PARENT_END]);
                        }
                        hunters[agent_idx][X_COORD_IDX] = incoming.move_request.x;
                        hunters[agent_idx][Y_COORD_IDX] = incoming.move_request.y;
                        map[hunters[agent_idx][X_COORD_IDX]][hunters[agent_idx][Y_COORD_IDX]] = 'H';
                    }
                    if(hunters[agent_idx][ENERGY_IDX] < 1){
                        hunters[agent_idx][ENERGY_IDX] = -1;
                        map[hunters[agent_idx][X_COORD_IDX]][hunters[agent_idx][Y_COORD_IDX]] = ' ';
                        --hunter_remaining;
                        kill(hunters[agent_idx][PID_IDX], SIGTERM);
                        waitpid(hunters[agent_idx][PID_IDX], NULL, WNOHANG);
                        close(pipefds[agent_idx][PARENT_END]);
                        mapUpdated = 1;
                        continue;
                    }
                    if(pfds[fd_idx+1].revents & POLLOUT){
                        pfds[fd_idx+1].revents = 0;
                        outgoing = getOutgoingMessageHunter(agent_idx);
                        fflush(0);
                        while(write(pipefds[agent_idx][PARENT_END], &outgoing, sizeof(outgoing)) != sizeof(outgoing))
                            fflush(0);
                    }
                }
            }
        }
        for(agent_idx=0; agent_idx<num_preys; ++agent_idx, fd_idx+=2){  // fd_idx continues from prev. loop.
            if(preys[agent_idx][ENERGY_IDX] == -1)
                continue;
            if(pfds[fd_idx].revents & POLLIN){
                pfds[fd_idx].revents = 0;
                readBytes = read(pipefds[num_hunters+agent_idx][PARENT_END], &incoming, sizeof(incoming));
                // printf("Prey\n");
                if(readBytes == sizeof(incoming)) {
                    if(isCollision(incoming.move_request.x, incoming.move_request.y, 'P'))
                        ;
                    else{
                        map[preys[agent_idx][X_COORD_IDX]][preys[agent_idx][Y_COORD_IDX]] = ' ';
                        mapUpdated = 1;
                        preys[agent_idx][X_COORD_IDX] = incoming.move_request.x;
                        preys[agent_idx][Y_COORD_IDX] = incoming.move_request.y;
                        if(map[incoming.move_request.x][incoming.move_request.y] == 'H'){
                            --prey_remaining;
                            int eatenIdx = findPrey(incoming.move_request.x, incoming.move_request.y);
                            tmp = preys[eatenIdx][ENERGY_IDX];
                            preys[eatenIdx][ENERGY_IDX] = -1;
                            kill(preys[eatenIdx][PID_IDX], SIGTERM);
                            waitpid(preys[eatenIdx][PID_IDX], NULL, WNOHANG);
                            close(pipefds[num_hunters+eatenIdx][PARENT_END]);
                            int eaterIdx = findHunter(incoming.move_request.x, incoming.move_request.y);
                            if(eaterIdx == -1)
                                map[preys[agent_idx][X_COORD_IDX]][preys[agent_idx][Y_COORD_IDX]] = 'P';
                            hunters[eaterIdx][ENERGY_IDX] += tmp;
                        }
                        else
                            map[preys[agent_idx][X_COORD_IDX]][preys[agent_idx][Y_COORD_IDX]] = 'P';
                    }
                    if(pfds[fd_idx+1].revents & POLLOUT){
                        pfds[fd_idx+1].revents = 0;
                        outgoing = getOutgoingMessagePrey(agent_idx);
                        while(write(pipefds[num_hunters+agent_idx][PARENT_END], &outgoing, sizeof(outgoing)) != sizeof(outgoing))
                            fflush(0);
                    }
                }
            }
        }
    }
    if(mapUpdated){
        printMap(width, height, map);
        mapUpdated = 0;
    }
    _beforeExit();
    return 0;
}