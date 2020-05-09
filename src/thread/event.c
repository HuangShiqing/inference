#include <stdlib.h>
#include <stdio.h>
#include <unistd.h> //getpid()
#include <string.h>

#include "event.h"

void event_init(event_parameter *event_par, int autoreset)
{
    // event_par = (event_parameter *)malloc(sizeof(event_parameter));
    event_par->mAutoReset = autoreset;          // 为1时退出阻塞后会重新准备下一次阻塞
    event_par->mQuery = 0;                      // 参数为0时会阻塞wait
    pthread_cond_init(&(event_par->mID), NULL); // 需要一个mutex配合
    pthread_mutex_init(&(event_par->mQueryMutex), NULL);
}

void event_wake(event_parameter *event_par)
{
    pthread_mutex_lock(&(event_par->mQueryMutex));
    event_par->mQuery = 1;
    pthread_cond_signal(&(event_par->mID));
    pthread_mutex_unlock(&(event_par->mQueryMutex));
}

int event_wait(event_parameter *event_par)
{
    pthread_mutex_lock(&(event_par->mQueryMutex)); // 为了保护下面的while条件的结果和wait结果是原子的，https://www.zhihu.com/question/24116967

    while (!event_par->mQuery)                                           // TODO:为了啥
        pthread_cond_wait(&(event_par->mID), &(event_par->mQueryMutex)); // 会先unlock，然后加入唤醒队列，允许别的线程进入wait，直到被唤醒后再lock

    if (event_par->mAutoReset) //自动使能准备下一次阻塞
        event_par->mQuery = 0;

    pthread_mutex_unlock(&(event_par->mQueryMutex));
    return 1;
}