#ifndef MARKET_H
#define MARKET_H

// ==========================================
// 你可以修改这里的结构体定义，但请保留成员变量名
// ==========================================

struct Candle {
    double high;
    double low;
    double close;
    long long vol;
    char _padding[32];
};



#endif