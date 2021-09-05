#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <iterator>

using namespace std;

vector<vector<int> > database;

class Item {
    vector<int> itemset;
    int support;
    int size;

    public:
    Item(vector<int> v, int s) : itemset(v), support(s), size(itemset.size()) {

    }
    Item(vector<int> v) : itemset(v), support(1), size(itemset.size()) {

    }
    Item() : itemset(vector<int>()), support(0), size(itemset.size()) {

    }
    int getSize() const {
        return size;
    }
    int getSupportCount() const {
        return support;
    }
    double getSupport() const {
        return double(support) / database.size();
    }
    vector<int> getItemSet() const {
        return itemset;
    }
    void addSupport() {
        support += 1;
    }
    bool operator==(const Item& rhs) const {
        return itemset == rhs.getItemSet();
    }
    bool operator!=(const Item& rhs) const {
        return !(*this == rhs);
    }
    bool operator<(const Item& rhs) const {
        return itemset < rhs.getItemSet();
        return 1;
    }
};

void printVector(vector<int>& v, ostream& ofs) {
    ostringstream oss;
    if(!v.empty()) {
        oss << "{";
        copy(v.begin(), v.end()-1,
        ostream_iterator<int>(oss, ","));
        oss << v.back();
        oss << "}\t";
    }
    ofs.write(oss.str().c_str(), oss.str().size());
}

Item unionItem(Item a, Item b) {
    vector<int> v(a.getSize() + b.getSize());
    vector<int>::iterator it;
    vector<int> v1 = a.getItemSet();
    vector<int> v2 = b.getItemSet();
    it = set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
    v.resize(it - v.begin());
    if(v.size() == v1.size() + 1) {
        sort(v.begin(), v.end());
        return Item(v, 0);
    } else {
        return Item();
    }
}


bool isSubset(Item a, Item b) {
    // a의 사이즈가 항상 작게 만들어줌
    if(a.getSize() > b.getSize()) {
        swap(a, b);
    }
    vector<int> v(a.getSize() + b.getSize());
    vector<int>::iterator it;
    vector<int> v1 = a.getItemSet();
    vector<int> v2 = b.getItemSet();
    it = set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
    v.resize(it - v.begin());
    if(v == v1) return true;
    else return false;
}

bool isSubset(Item a, vector<int> v2) {
    vector<int> v(a.getSize() + v2.size());
    vector<int>::iterator it;
    vector<int> v1 = a.getItemSet();
    it = set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
    v.resize(it - v.begin());
    if(v == v1) return true;
    else return false;
}

void seperate(set<Item>& sI, vector<int>& total, vector<int>& sub, ostream& ofs) {
    if(sub.empty() || total == sub) return;
    vector<int> v(total.size() + sub.size());
    vector<int>::iterator it;
    it = set_difference(total.begin(), total.end(), sub.begin(), sub.end(), v.begin());
    v.resize(it - v.begin());
    printVector(v, ofs);
    printVector(sub, ofs);
    char p1[10], p2[10];
    sprintf(p1, "%.2f\t",sI.find(total)->getSupport()*100);
    ofs.write(p1, strlen(p1));
    int u = sI.find(total)->getSupportCount();
    int d = sI.find(v)->getSupportCount();
    sprintf(p2, "%.2f\n",(double)u / d * 100);
    ofs.write(p2, strlen(p2));

}

void search(set<Item>& sI, vector<int>& total, vector<int>& subset, int k, ostream& ofs) {
    if(k == total.size()) {
        seperate(sI, total, subset, ofs);
        return;
    } else {
        subset.push_back(total[k]);
        search(sI, total, subset, k+1, ofs);
        subset.pop_back();
        search(sI, total, subset, k+1, ofs);
    }
}

void firstScan(double min_support, vector<Item>& v, set<Item>& sI) {
    set<int> s;
    for(int i = 0; i < database.size(); ++i) {
        for(int j = 0; j < database[i].size(); ++j) {
            int f = database[i][j];
            vector<int> one_len_v = vector<int>(1, f);
            Item newItem(one_len_v);
            if(s.find(f) == s.end()) {
                s.insert(f);
                sI.insert(newItem);
            } else {
                auto it = sI.find(newItem);
                int now_sup = (*it).getSupportCount();
                sI.erase(newItem);
                sI.insert(Item(one_len_v, now_sup+1));
            }
        }
    }
    for(auto it = sI.begin(); it != sI.end(); ++it) {
        v.push_back(*it);
    }
}

void apriori(double min_support, ofstream& ofs) {
    int idx = 1;
    vector<Item> L;
    set<Item> sI;
    firstScan(min_support, L, sI);
    for(int k = 1; !L.empty(); ++k) {
        vector<Item> nextL;
        multimap<Item, int> mm;
        for(int m = 0; m < L.size(); ++m) {
            for(int n = m+1; n < L.size(); ++n) {
                Item newItem = unionItem(L[m],L[n]);
                vector<int > nv = newItem.getItemSet();
                if(newItem.getSize() == k+1) {
                    multimap<Item, int>::iterator it = mm.find(newItem);
                    if(it == mm.end()) {
                        mm.insert(make_pair(newItem, 1));
                    } else {
                        it->second += 1;
                    }
                } else {
                    continue;
                }
            }
        }
        for(auto it = mm.begin(); it != mm.end(); ++it) {
            if(it->second == (k*(k+1))/2) {
                nextL.push_back(it->first);
            }
        }
        for(int dbIdx = 0; dbIdx < database.size(); ++dbIdx) {
            for(auto it = nextL.begin(); it != nextL.end(); ++it) {
                Item* comp = &(*it);
                if(isSubset(*comp, database[dbIdx])) {
                    comp->addSupport();
                }
            }
        }
        for(auto it = nextL.begin(); it != nextL.end();) {
            if(it->getSupport() >= min_support) {
                sI.insert(*it);
                it++;
            } else {
                it = nextL.erase(it);
            }
        }
        for(auto it = nextL.begin(); it != nextL.end(); ++it) {
            Item item = *it;
            vector<int> total = item.getItemSet();
            vector<int> subset;
            search(sI, total, subset, 0, ofs);
        }
        L = nextL;
    }
}

int main(int argc, char* argv[]) {
    double min_support;
    string input_name, output_name;
    ifstream ifs;
    ofstream ofs;
    if(argc != 4) {
        cout << "Usage: " << argv[0] << " min_support input_file output_file\n";
        return 1;
    }
    min_support = stod(argv[1]);
    input_name = argv[2];
    output_name = argv[3];
    min_support /= 100;
    ifs.open(input_name);
    if(ifs.is_open()) {
        string line;
        while(getline(ifs, line)) {
            vector<int> trx;
            int now = 0;
            for(int pos = 0; pos < line.size(); ++pos) {
                if(line[pos] == '\t') {
                    trx.push_back(stoi(line.substr(now, pos-now+1)));
                    now = pos+1;
                }
            }
            trx.push_back(stoi(line.substr(now)));
            sort(trx.begin(), trx.end());
            database.push_back(trx);
        }
        ifs.close();
    }
    ofs.open(output_name);
    apriori(min_support, ofs);
    ofs.close();
}
