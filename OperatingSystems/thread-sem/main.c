#include <stdlib.h>
#include "do_not_submit.h"
#include <pthread.h>
#include <semaphore.h>

#define ANT_NULL -1000

pthread_mutex_t locks[GRIDSIZE * GRIDSIZE];
pthread_mutex_t sleepMutex;
pthread_mutex_t drawMutex;
pthread_cond_t sleepCond = PTHREAD_COND_INITIALIZER;
bool exitSignal;

typedef struct neighborCells
{
  int foodWidthOffset;
  int foodHeightOffset;
  int emptyWidthOffset;
  int emptyHeightOffset;
  int emptyCell;
} nCells;
typedef struct antStatus
{
  int status;
  int width;
  int height;
  char symbol;
  char placeholderSymbol;
  int id;
} antStatus;

void ant_moveToLoc(antStatus *ant, int i, int j) {
  //0,0 and 29,29 are valid locations.
  pthread_mutex_lock(&drawMutex);
  putCharTo(i, j, ant->symbol);
  putCharTo(ant->width, ant->height, ant->placeholderSymbol);
  pthread_mutex_unlock(&drawMutex);
  ant->width = i;
  ant->height = j;
}

void ant_state0Action(antStatus *ant) {
  int i, j, mutexIndex, emptyCellExists=0;
  char tmp;
  for(i=ant->width-1; i<=ant->width+1; i++) {
    if(i >= GRIDSIZE || i < 0)
      continue;
    for(j=ant->height-1; j<=ant->height+1; j++) {
      if(j >= GRIDSIZE || j < 0)
        continue;
      mutexIndex = i + j*GRIDSIZE;
      pthread_mutex_lock(&locks[mutexIndex]);
      tmp = lookCharAt(i, j);
      if(tmp == 'o') {
        ant->status = 1;
        ant->symbol = 'P';
        ant->placeholderSymbol = '-';
        ant_moveToLoc(ant, i, j);
        pthread_mutex_unlock(&locks[mutexIndex]);
        return;
      }
      else if(tmp == '-') {
        emptyCellExists++;
      }
      pthread_mutex_unlock(&locks[mutexIndex]);
    }
  }
  if(ant->symbol == '1' && emptyCellExists) {
    int x, y;
    while(emptyCellExists) {
      int i = rand() % 8;
      x = ant->width-1; y = ant->height-1;
      if(i==0 && (x >= 0 && y >= 0)) {
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==1 && (ant->height-1 >= 0)) {
        x = ant->width; y = ant->height-1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==2 && (ant->width+1 < GRIDSIZE && ant->height-1 >= 0)) {
        x = ant->width+1; y = ant->height-1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==3 && (ant->width-1 >= 0)) {
        x = ant->width-1; y = ant->height;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==4 && (ant->width+1 < GRIDSIZE)) {
        x = ant->width+1; y = ant->height;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==5 && (ant->width-1 >= 0 && ant->height+1 < GRIDSIZE)) {
        x = ant->width-1; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==6 && (ant->height+1 < GRIDSIZE)) {
        x = ant->width; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==7 && (ant->width+1 < GRIDSIZE && ant->height+1 < GRIDSIZE)) {
        x = ant->width+1; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else
        continue;
    }
  }
}

void ant_state1Action(antStatus *ant) {
  int i, j, mutexIndex, emptyCellExists=0, foodInNeighbor=0;
  char tmp;
  for(i=ant->width-1; i<=ant->width+1; i++) {
    if(i >= GRIDSIZE || i < 0)
      continue;
    for(j=ant->height-1; j<=ant->height+1; j++) {
      if(j >= GRIDSIZE || j < 0)
        continue;
      mutexIndex = i + j*GRIDSIZE;
      pthread_mutex_lock(&locks[mutexIndex]);
      tmp = lookCharAt(i, j);
      if(tmp == 'o') {
        foodInNeighbor = 1;
      }
      else if(tmp == '-') {
        emptyCellExists++;
      }
      pthread_mutex_unlock(&locks[mutexIndex]);
    }
  }
  if(emptyCellExists) {
    int x, y;
    if(foodInNeighbor) {
      ant->symbol = '1';
      ant->placeholderSymbol = 'o';
      ant->status = 2;
    }
    while(emptyCellExists > 0) {
      i = rand() % 8;
      x = ant->width-1; y = ant->height-1;
      if(i==0 && (x >= 0 && y >= 0)) {
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==1 && (ant->height-1 >= 0)) {
        x = ant->width; y = ant->height-1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==2 && (ant->width+1 < GRIDSIZE && ant->height-1 >= 0)) {
        x = ant->width+1; y = ant->height-1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==3 && (ant->width-1 >= 0)) {
        x = ant->width-1; y = ant->height;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==4 && (ant->width+1 < GRIDSIZE)) {
        x = ant->width+1; y = ant->height;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==5 && (ant->width-1 >= 0 && ant->height+1 < GRIDSIZE)) {
        x = ant->width-1; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==6 && (ant->height+1 < GRIDSIZE)) {
        x = ant->width; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==7 && (ant->width+1 < GRIDSIZE && ant->height+1 < GRIDSIZE)) {
        x = ant->width+1; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else
        continue;
    }
    if(emptyCellExists <= 0) {
      ant->symbol = 'P';
      ant->placeholderSymbol = '-';
      ant->status = 1;
    } else
      ant->placeholderSymbol = '-';
  }
}

void ant_state2Action(antStatus *ant) {
  int i, j, x, y, mutexIndex, emptyCellExists=0;
  char tmp;
  for(i=ant->width-1; i<=ant->width+1; i++) {
    if(i >= GRIDSIZE || i < 0)
      continue;
    for(j=ant->height-1; j<=ant->height+1; j++) {
      if(j >= GRIDSIZE || j < 0)
        continue;
      mutexIndex = i + j*GRIDSIZE;
      pthread_mutex_lock(&locks[mutexIndex]);
      tmp = lookCharAt(i, j);
      if(tmp == '-') {
        emptyCellExists++;
      }
      pthread_mutex_unlock(&locks[mutexIndex]);
    }
  }
  while(emptyCellExists) {
      int i = rand() % 8;
      x = ant->width-1; y = ant->height-1;
      if(i==0 && (x >= 0 && y >= 0)) {
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==1 && (ant->height-1 >= 0)) {
        x = ant->width; y = ant->height-1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==2 && (ant->width+1 < GRIDSIZE && ant->height-1 >= 0)) {
        x = ant->width+1; y = ant->height-1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==3 && (ant->width-1 >= 0)) {
        x = ant->width-1; y = ant->height;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==4 && (ant->width+1 < GRIDSIZE)) {
        x = ant->width+1; y = ant->height;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==5 && (ant->width-1 >= 0 && ant->height+1 < GRIDSIZE)) {
        x = ant->width-1; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==6 && (ant->height+1 < GRIDSIZE)) {
        x = ant->width; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else if(i==7 && (ant->width+1 < GRIDSIZE && ant->height+1 < GRIDSIZE)) {
        x = ant->width+1; y = ant->height+1;
        mutexIndex = x + y*GRIDSIZE;
        pthread_mutex_lock(&locks[mutexIndex]);
        if (lookCharAt(x, y) == '-') {
          ant_moveToLoc(ant, x, y);
          pthread_mutex_unlock(&locks[mutexIndex]);
          break;
        }
        else {
          emptyCellExists--;
        }
        pthread_mutex_unlock(&locks[mutexIndex]);
      }
      else
        continue;
    }
  if(emptyCellExists) {
    ant->status = 0;
    ant->placeholderSymbol = '-';
  }
}


void *antRoutine(void* coord) {
  antStatus* ant = (antStatus*) coord;
  ant->placeholderSymbol = '-';
  
  ////////////////////////////
  // status 0: without food //
  // status 1: with food    //
  // status 2: tired        //
  // status 3: sleeping     //
  ////////////////////////////
  
  while(TRUE) {
    if(exitSignal)
      break;
      
    // STATE3
    pthread_mutex_lock(&sleepMutex);
    while(getSleeperN() > ant->id) {
      pthread_cond_wait(&sleepCond, &sleepMutex);
      if (getSleeperN() > ant->id)
        pthread_cond_signal(&sleepCond);
    }
    pthread_mutex_unlock(&sleepMutex);
    // STATE3
    
    switch(ant->status) {
      case 0: ant_state0Action(ant); break;
      case 1: ant_state1Action(ant); break;
      case 2: ant_state2Action(ant); break;
      // case 3: ant_state3Action(ant); break;
    }
    usleep(getDelay() * 1000 + (rand() % 10000));
  }
  free(coord);
  return NULL;
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  exitSignal = FALSE;
  
  antStatus* antArg;
  int numAnts, numFoods, maxSimTime;
  pthread_t* ants;
  int antIndex = 0;
  int i,j;
  int a,b;
  
  sscanf(argv[1], "%i", &numAnts);
  sscanf(argv[2], "%i", &numFoods);
  sscanf(argv[3], "%i", &maxSimTime);
  maxSimTime *= 1000*1000; // To seconds.
  ants = malloc(numAnts*sizeof(pthread_t));
  
  for (i = 0; i < GRIDSIZE; i++) {
    for (j = 0; j < GRIDSIZE; j++) {
      putCharTo(i, j, '-');
      pthread_mutex_init(&locks[i + j*GRIDSIZE], NULL);
    }
  }
  pthread_mutex_init(&sleepMutex, NULL);
  pthread_mutex_init(&drawMutex, NULL);
  for (i = 0; i < numAnts; i++) {
    do {
      a = rand() % GRIDSIZE;
      b = rand() % GRIDSIZE;
    }while (lookCharAt(a,b) != '-');
    putCharTo(a, b, '1');
    antArg = malloc(sizeof(antStatus));
    antArg->width = a;
    antArg->height = b;
    antArg->status = 0;
    antArg->symbol = '1';
    antArg->id = antIndex;
    pthread_create(&ants[antIndex++], NULL, antRoutine, antArg);
  }
  for (i = 0; i < numFoods; i++) {
    do {
      a = rand() % GRIDSIZE;
      b = rand() % GRIDSIZE;
    }while (lookCharAt(a,b) != '-');
    putCharTo(a, b, 'o');
  }

  startCurses();

  char c;
  while (maxSimTime > 0) {
    pthread_mutex_lock(&drawMutex);
    drawWindow();
    pthread_mutex_unlock(&drawMutex);
    c = 0;
    c = getch();
    if (c == 'q' || c == ESC)
      break;
    if (c == '+') {
      setDelay(getDelay()+10);
    }
    if (c == '-') {
      setDelay(getDelay()-10);
    }
    if (c == '*') {
      if(getSleeperN() < numAnts)
        setSleeperN(getSleeperN()+1);
    }
    if (c == '/') {
      if(getSleeperN() > 0)
        setSleeperN(getSleeperN()-1);
      pthread_cond_signal(&sleepCond);
    }
    usleep(DRAWDELAY);
    maxSimTime -= DRAWDELAY;
  }
  exitSignal = TRUE;
  // do not forget freeing the resources you get
  endCurses();
  while(getSleeperN() > 0) {
    setSleeperN(getSleeperN()-1);
      pthread_cond_signal(&sleepCond);
  }
  pthread_cond_destroy(&sleepCond);
  pthread_mutex_destroy(&sleepMutex);
  for(i=0; i<antIndex; i++)
    pthread_join(ants[i], NULL);
  for (i = 0; i < GRIDSIZE; i++) {
    for (j = 0; j < GRIDSIZE; j++) {
      // pthread_mutex_unlock(&(locks[i + j*GRIDSIZE]));
      pthread_mutex_destroy(&(locks[i + j*GRIDSIZE]));
    }
  }
  free(ants);
  return 0;
}
