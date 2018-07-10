
#include "ll.hpp"

namespace ll{namespace ui{

struct mulchar{
    mulchar(char c, int num): c_(c), count_(num){}
    char c_;
    int count_;
};
std::ostream& operator<<(std::ostream& os, const mulchar& dc){
    LL_REPEAT(dc.count_) os<<dc.c_;
    return os;
}

class board{
public:
    board(int w, int h): width_(w), height_(h){}
    // LL_BAN_COPY(board);

    int width() const{ return width_; }
    int height() const{ return height_; }

    void clear(){ umapdots_.clear(); }
    void set(int x, int y, char c='+'){
        umapdots_[pos2idx(x, y)]    =   c;
    }

    void print(std::ostream& os=std::cout){
        for(int y=0; y<height_; ++y){
            os<<"|";
            for(int x=0; x<width_; ++x){
                auto iter   =   umapdots_.find(pos2idx(x, y));
                os<<(iter==umapdots_.end()? ' ': iter->second);
            }
            os<<"\n";
        }
        os<<"+"<<mulchar('-', width_)<<"\n";
    }

private:
    int width_, height_;
    // sparse
    std::unordered_map<int, char> umapdots_;

    inline int pos2idx(int x, int y) const{ return y*width_+x; }
};

template<typename T1=double, typename T2=double>
board& plot(board& b, const std::vector<T1>& xs, const std::vector<T2>& ys, 
    char mk='o'){
    assert(xs.size()==ys.size() && "bad inputs");

    T1 minx, maxx;
    T2 miny, maxy;
    {
        auto prx    =   std::minmax_element(xs.begin(), xs.end());
        minx    =   *prx.first;
        maxx    =   *prx.second;     
        auto pry    =   std::minmax_element(ys.begin(), ys.end());
        miny    =   *pry.first;
        maxy    =   *pry.second;
    }

    for(int i=0; i<xs.size(); ++i){
        int x   =   (xs[i]-minx)/(maxx-minx)* (b.width()-1);
        int y   =   (ys[i]-miny)/(maxy-miny)* (b.height()-1);
        b.set(x, y, mk);
    }

    return b;
}

template<typename T>
board& plot(board& b, const std::vector<T>& vals){
    return plot<T, T>(b, ll::range<T>(vals.size()), vals);
}

}}
