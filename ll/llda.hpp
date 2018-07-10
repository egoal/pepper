#include "ll.hpp"

namespace ll{ namespace da{

enum class cmp{
    less, equal, greater
};

template<typename T=double>
T mean(const std::vector<T>& vals){
    return std::accumulate(vals.begin(), vals.end(), T(0))/vals.size();
}

template<typename T=double>
std::tuple<T, T> mean_and_variance(const std::vector<T>& vals){
    auto m  =   mean<T>(vals);
    auto v  =   std::accumulate(vals.begin(), vals.end(), T(0), 
        [m](double sum, double v){ return sum+(v-m)*(v-m); })/vals.size();

    return std::make_tuple(m, v);
}

template<typename T=double>
std::vector<T> percent_quantile(const std::vector<T>& c, std::initializer_list<double> ratios){
    double mxrt =   *(std::max_element(ratios.begin(), ratios.end()));
    assert(mxrt<=1. && "bad range");

    auto mIdx   =   static_cast<std::size_t>(c.size()*mxrt);
    auto sc =   c;
    std::partial_sort(sc.begin(), sc.begin()+mIdx+1, std::end(sc));

    std::vector<T> vec(ratios.size());
    std::transform(ratios.begin(), ratios.end(), vec.begin(), [&](double rt){ 
        return *(std::begin(sc)+ static_cast<std::size_t>(sc.size()*rt)); });

    return vec;
}

template<typename BOP> std::vector<std::tuple<int, int, double > > match_bf(
    int maxidx0, int maxidx1, BOP bop, bool crossCheck=false){
    assert(maxidx0>0 && maxidx1>0);
    
    std::vector<std::tuple<int, int, double > > matches;

    auto rng0   =   ll::range(maxidx0);
    auto rng1   =   ll::range(maxidx1);
    for(auto& i: rng0){
        int minidx  =   -1;
        double mindis   =   std::numeric_limits<double>::max();
        for(auto& j: rng1){
            double dis  =   bop(i, j);
            if(dis<mindis){
                mindis  =   dis;
                minidx  =   j;
            }
        }

        if(minidx<0) continue;

        if(crossCheck && std::any_of(rng0.begin(), rng0.end(), 
            [&](int i){ return bop(i, minidx)<mindis; }))
            continue;

        matches.push_back(std::make_tuple(i, minidx, mindis));
    }

    return matches;
}

template<typename BOP> std::vector<std::vector<std::tuple<int, int, double > > >
    match_knn(int maxidx0, int maxidx1, BOP bop, int k){
    assert(k>=1 && "bad arguments");
    using thematch  =   std::tuple<int, int, double >;
    std::vector<std::vector<thematch > > matches;

    if(k==1){
        LL_LOG<<"better to use match_bf instead\n";
        auto dmatches   =   match_bf<BOP>(maxidx0, maxidx1, bop);
        matches.reserve(matches.size());
        std::transform(dmatches.begin(), dmatches.end(), matches.begin(), 
            [](const thematch& m){ 
                std::vector<thematch > ms;
                ms.push_back(m);
            });
        return matches;
    }


}

// p-> a-b
double distance_to_line(double ax, double ay, double bx, double by, double px, double py){
    double apx(px-ax), apy(py-ay), abx(bx-ax), aby(by-ay);
    assert(abx*abx+aby*aby> 1e-8 && "cannot use same point");

    double dis  =   (apx*aby- apy*abx)/std::sqrt(abx*abx+ aby*aby);
    return LL_ABS(dis);
}

}}