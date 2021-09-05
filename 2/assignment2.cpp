#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <iterator>
#include <cmath>

using namespace std;

vector<vector<int> > db_table;
vector<string> attribute_name;
vector<map<string, int> > Word2Idx;
vector<map<int, string> > Idx2Word;
vector<int> category_size;
int attr_size = 0;
int test_attr_size = 0;

vector<vector<int> > test_db_table;
vector<int> result;

class decisionTree {
    public:
        vector<vector<int> > table;
        vector<decisionTree> childs;
        vector<int> hashTable;
        int test_attribute;
        int predict;
        int height;
        bool leaf;

    public:
        decisionTree(vector<vector<int> > t) : table(t), childs(vector<decisionTree> ()), test_attribute(-1), predict(-1), height(0), hashTable(vector<int>()), leaf(false) {}

        decisionTree() {}
        void addChild(decisionTree t) {
            childs.push_back(t);
        }

        vector<vector<int> > getTable() {
            return table;
        }

        void setTestAttribute(int k) {
            test_attribute = k;
        }
        
        int getTableLines() {
            return table.size();
        }

        bool isEmpty() {
            return table == vector<vector<int> >();
        }

        void setHashTable(vector<int>& v) {
            hashTable.clear();
            hashTable.assign(v.size(), 0);
            copy(v.begin(), v.end(), hashTable.begin());
        }
        
};

double infoGain(int c, int total) {
    if(c == 0 || total == 0) return 0;
    double x = (double)c / total;
    double ret = -(x * log2(x));
    return ret;
}

double splitInfo(vector<vector<int> >& count_table, int total, int attr_idx) {
    double ret = 0.0;
    vector<int> count(category_size[attr_idx], 0);
    for(int i = 0; i < count_table.size(); ++i) {
        for(int j = 0; j < count_table[i].size(); ++j) {
            count[i] += count_table[i][j];
        }
    }
    for(int i = 0; i < count.size(); ++i) {
        ret += infoGain(count[i], total);
    }
    return ret;
}

double getGini(vector<vector<int> >& count_table, vector<int>& v, int tot, int size) {
    double ret = 1.0;
    vector<int> vec(category_size[attr_size-1], 0);
    for(int i = 0; i < v.size(); ++i) {
        int cnt = 0;
        for(int j = 0; j < vec.size(); ++j) {
            vec[j] += count_table[v[i]][j];
        }
    }
    for(int i = 0; i < vec.size(); ++i) {
        ret -= ((double)vec[i] / tot) * ((double)vec[i] / tot);
    }
    return ret;
}

double search(vector<vector<int> >& count_table, vector<int> allAttr, int pos, vector<int>& subset, vector<int>& part, double& minVal) {
    if(pos == allAttr.size()) {
        if(subset.size() == 0 || subset.size() == allAttr.size() /*|| subset.size() < allAttr.size() / 2*/) {
            return 1e9;
        } else {
            vector<int> v(allAttr.size() + subset.size());
            vector<int>::iterator it;
            it = set_difference(allAttr.begin(), allAttr.end(), subset.begin(), subset.end(), v.begin());
            v.resize(it - v.begin());
            int cnt1 = 0, cnt2 = 0;
            for(int i = 0; i < v.size(); ++i) {
                for(int j = 0; j < count_table[v[i]].size(); ++j) {
                    cnt1 += count_table[v[i]][j];
                }
            }
            for(int i = 0; i < subset.size(); ++i) {
                for(int j = 0; j < count_table[subset[i]].size(); ++j) {
                    cnt2 += count_table[subset[i]][j];
                }
            }
            int total = cnt1 + cnt2;
            double ret = (double)cnt1 / total * getGini(count_table, v, cnt1, allAttr.size()) + (double)cnt2 / total * getGini(count_table, subset, cnt2, allAttr.size());
            if(minVal > ret) {
                minVal = ret;
                part.clear();
                part.assign(subset.size(), 0);
                copy(subset.begin(), subset.end(), part.begin());
            }
            return ret;
        }
    }
    subset.push_back(allAttr[pos]);
    search(count_table, allAttr, pos+1, subset, part, minVal);
    subset.pop_back();
    search(count_table, allAttr, pos+1, subset, part, minVal);
    return minVal;
}

double binarySplit(vector<vector<int> >& count_table, int total, int attr_idx, vector<int>& part) {
    vector<int> allAttr;
    vector<int> subset;
    set<int> s;
    for(int i = 0; i < count_table.size() ;++i) {
        for(int j = 0; j < count_table[i].size(); ++j) {
            if(count_table[i][j] > 0) {
                s.insert(i);
            }
        }
    }
    for(auto it = s.begin(); it != s.end(); ++it) {
        allAttr.push_back(*it);
    }
    double minVal = 1e9;
    double ret = search(count_table, allAttr, 0, subset, part, minVal);
    return ret;
}

double gini(vector<vector<int> >& count_table, int total, int attr_idx, vector<int>& part) {
    double ret = 0.0;
    ret = binarySplit(count_table, total, attr_idx, part);
    return ret;
}

double getInfoD(vector<int>& count, int total) {
    int size = count.size();
    double ret = 0;
    for(int i = 0; i < count.size(); ++i) {
        if(count[i] == 0) continue;
        ret += infoGain(count[i], total);
    }
    return ret;
}

double getInfoGain(vector<vector<int> >& count_table, int total, int idx, int type) {
    double ret = 0.0;
    vector<int> sum_cols(category_size[idx], 0);
    for(int i = 0; i < count_table.size(); ++i) {
        for(int j = 0; j < count_table[i].size(); ++j) {
            sum_cols[i] += count_table[i][j];
        }
    }
    vector<int> sum_rows(category_size[attr_size-1], 0);
    for(int i = 0; i < count_table.size(); ++i) {
        for(int j = 0; j < count_table[i].size(); ++j) {
            sum_rows[j] += count_table[i][j];
        }
    }


    double InfoD = getInfoD(sum_rows, total);
    for(int i = 0; i < count_table.size(); ++i) {
        double tmp = 0.0;
        for(int j = 0; j < count_table[i].size(); ++j) {
            tmp += infoGain(count_table[i][j], sum_cols[i]);
        }
        tmp *= (double)(sum_cols[i]) / total;
        ret += tmp;
    }
    ret = InfoD - ret;
    if(type == 1) {
        double split = splitInfo(count_table, total, idx);
        ret /= split;
    }
    return ret;
}

int findNextAttrIdx(vector<vector<int> >& table, vector<bool>& use, int type) {
    int ret_idx = -1;
    double cost = 0;
    
    vector<vector<vector<int> > > count_table;
    for(int i = 0; i < attr_size-1; ++i) {
        vector<vector<int> > tmp(category_size[i], vector<int>(category_size[attr_size-1], 0));
        for(int j = 0; j < table.size(); ++j) {
            int category = table[j][i];
            int label = table[j][attr_size-1];
            tmp[category][label] += 1;
        }
        count_table.push_back(tmp);
    }

    for(int i = 0; i < test_attr_size; ++i) {
        if(use[i]) continue;
        double curr_cost = getInfoGain(count_table[i], table.size(), i, type);
        if(curr_cost > cost) {
            cost = curr_cost;
            ret_idx = i;
        }
    }
    return ret_idx;
}

int findNextAttrIdxGini(vector<vector<int> >& table, vector<int>& sep) {
    int ret_idx = -1;
    double cost = 1e9;
    
    vector<vector<vector<int> > > count_table;
    for(int i = 0; i < attr_size-1; ++i) {
        vector<vector<int> > tmp(category_size[i], vector<int>(category_size[attr_size-1], 0));
        for(int j = 0; j < table.size(); ++j) {
            int category = table[j][i];
            int label = table[j][attr_size-1];
            tmp[category][label] += 1;
        }
        count_table.push_back(tmp);
    }

    for(int i = 0; i < test_attr_size; ++i) {
        vector<int> part;
        double curr_cost = gini(count_table[i], table.size(), i, part);
        if(curr_cost < cost) {
            sep.clear();
            sep.assign(part.size(), 0);
            copy(part.begin(), part.end(), sep.begin());
            cost = curr_cost;
            ret_idx = i;
        }
    }
    return ret_idx;
}

void buildTree(decisionTree& t, int attr_idx, vector<bool>& use, int type) {
    if(t.leaf == true) {
        return;
    }
    vector<vector<int> > table = t.getTable();
    if(table == vector<vector<int> >()) {
        return;
    }
    int next_test_attr = findNextAttrIdx(table, use, type);
    t.setTestAttribute(next_test_attr);
    if(next_test_attr == -1) {
        vector<int> counts(category_size[attr_size-1], 0);
        int max_idx = -1;
        int max_val = 0;
        for(int i = 0; i < table.size(); ++i) {
            int result_idx = table[i][attr_size-1];
            counts[result_idx] += 1;
            if(counts[result_idx] > max_val) {
                max_val = counts[result_idx];
                max_idx = result_idx;
            }
        }
        t.predict = max_idx;
        t.leaf = true;
        return;
    }
    use[next_test_attr] = true;
    vector<vector<vector<int> > > total_table(category_size[next_test_attr], vector<vector<int> >());
    for(int i = 0; i < table.size(); ++i) {
        vector<int> line = table[i];
        int number = line[next_test_attr];
        total_table[number].push_back(line);
    }
    for(int i = 0; i < category_size[next_test_attr]; ++i) {
        t.addChild(decisionTree(total_table[i]));
        t.childs[i].height = t.height + 1;
    }
    for(int i = 0; i < total_table.size(); ++i) {
        int cnt = 0;
        for(int j = 0; j < total_table[i].size(); ++j) {
            if(total_table[i][j][attr_size-1] == total_table[i][0][attr_size-1]) {
                cnt += 1;
            }
        }
        if(total_table[i].size() != 0 && cnt == total_table[i].size()) {
            t.childs[i].predict = total_table[i][0][attr_size-1];
            t.childs[i].leaf = true;
        }
    }

    int majority_idx = -1;
    int majority_val = 0;
    vector<int> counts2(category_size[attr_size-1], 0);
    for(int i = 0; i < total_table.size(); ++i) {
        if(total_table[i].size() == 0) continue;
        for(int j = 0; j < total_table[i].size(); ++j) {
            counts2[total_table[i][j][attr_size-1]] += 1;
            if(majority_val < counts2[total_table[i][j][attr_size-1]]) {
                majority_val = counts2[total_table[i][j][attr_size-1]];
                majority_idx = total_table[i][j][attr_size-1];
            }
        }
    }

    t.predict = majority_idx;

    for(int i = 0; i < total_table.size(); ++i) {
        if(total_table[i].size() == 0) {
            t.childs[i].predict = majority_idx;
            t.childs[i].leaf = true;
        }
    }

    for(int i = 0; i < category_size[next_test_attr]; ++i) {
        buildTree(t.childs[i], next_test_attr, use, type);
        use[next_test_attr] = false;
    }
}

void buildTreeGini(decisionTree& t, int attr_idx) {
    if(t.leaf == true) {
        return;
    }
    vector<vector<int> > table = t.getTable();
    if(table == vector<vector<int> >()) {
        return;
    }
    vector<int> sep;
    int next_test_attr = findNextAttrIdxGini(table, sep);
    t.setTestAttribute(next_test_attr);
    if(next_test_attr == -1) {
        vector<int> counts(category_size[attr_size-1], 0);
        int max_idx = -1;
        int max_val = 0;
        for(int i = 0; i < table.size(); ++i) {
            int result_idx = table[i][attr_size-1];
            counts[result_idx] += 1;
            if(counts[result_idx] > max_val) {
                max_val = counts[result_idx];
                max_idx = result_idx;
            }
        }
        cout<<endl;
        t.predict = max_idx;
        t.leaf = true;
        return;
    }
    vector<int> hashTable(category_size[next_test_attr], 0);
    for(int i = 0; i < sep.size(); ++i) {
        hashTable[sep[i]] = 1;
    }
    t.setHashTable(hashTable);
    vector<vector<vector<int> > > total_table(2, vector<vector<int> >());
    for(int i = 0; i < table.size(); ++i) {
        vector<int> line = table[i];
        int number = line[next_test_attr];
        total_table[hashTable[number]].push_back(line);
    }
    for(int i = 0; i < 2; ++i) {
        t.addChild(decisionTree(total_table[i]));
        t.childs[i].height = t.height + 1;
    }
    for(int i = 0; i < 2; ++i) {
        int cnt = 0;
        for(int j = 0; j < total_table[i].size(); ++j) {
            if(total_table[i][j][attr_size-1] == total_table[i][0][attr_size-1]) {
                cnt += 1;
            }
        }
        if(total_table[i].size() != 0 && cnt == total_table[i].size()) {
            t.childs[i].predict = total_table[i][0][attr_size-1];
            t.childs[i].leaf = true;
        }
    }

    int majority_idx = -1;
    int majority_val = 0;
    vector<int> counts2(category_size[attr_size-1], 0);
    for(int i = 0; i < total_table.size(); ++i) {
        if(total_table[i].size() == 0) continue;
        for(int j = 0; j < total_table[i].size(); ++j) {
            counts2[total_table[i][j][attr_size-1]] += 1;
            if(majority_val < counts2[total_table[i][j][attr_size-1]]) {
                majority_val = counts2[total_table[i][j][attr_size-1]];
                majority_idx = total_table[i][j][attr_size-1];
            }
        }
    }

    t.predict = majority_idx;

    for(int i = 0; i < total_table.size(); ++i) {
        if(total_table[i].size() == 0) {
            t.childs[i].predict = majority_idx;
            t.childs[i].leaf = true;
        }
    }

    for(int i = 0; i < 2; ++i) {
        buildTreeGini(t.childs[i], next_test_attr);
    }
}

decisionTree makeDecisionTree(int type) {
    vector<bool> use(test_attr_size, false); 
    decisionTree tree(db_table);
    if(type < 2)
        buildTree(tree, -1, use, type);
    else
        buildTreeGini(tree, -1);
    return tree;
}

int getPredictedIdx(decisionTree& t, vector<int>& line, int type) {
    if(t.leaf == true) return t.predict;
    int ret = -1;
    if(type < 2) {
        ret = getPredictedIdx(t.childs[line[t.test_attribute]], line, type);
    } else {
        ret = getPredictedIdx(t.childs[t.hashTable[line[t.test_attribute]]], line, type);
    }
    return ret;
}

void read_trainData(ifstream& ifs, string& train_name) {
    ifs.open(train_name);
    if(ifs.is_open()) {
        string line;
        getline(ifs, line);
        int now = 0;
        for(int pos = 0; pos < line.size(); ++pos) {
            if(line[pos] == '\t') {
                string value = line.substr(now, pos-now);
                attribute_name.push_back(value);
                attr_size += 1;
                now = pos+1;
            }
        }
        string value = line.substr(now);
        value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
        value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
        attribute_name.push_back(value);
        attr_size += 1;


        db_table.clear();
        Word2Idx.assign(attr_size, map<string, int>());
        Idx2Word.assign(attr_size, map<int, string>());
        category_size.assign(attr_size, 0);
        test_attr_size = attr_size - 1;

        int line_count = 0;
        while(getline(ifs, line)) {
            db_table.push_back(vector<int>(attr_size, -1));
            now = 0;
            int idx = 0;
            for(int pos = 0; pos < line.size(); ++pos) {
                if(line[pos] == '\t') {
                    string value = line.substr(now, pos-now);
                    if(Word2Idx[idx].find(value) != Word2Idx[idx].end()) {
                        db_table[line_count][idx] = Word2Idx[idx][value];
                    } else {
                        Word2Idx[idx][value] = Word2Idx[idx].size();
                        db_table[line_count][idx] = Word2Idx[idx][value];
                        Idx2Word[idx][Word2Idx[idx].size() - 1] = value;
                    }
                    now = pos+1;
                    idx += 1;
                }
            }
            string value = line.substr(now);
            value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
            value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());

            if(Word2Idx[idx].find(value) != Word2Idx[idx].end()) {
                db_table[line_count][idx] = Word2Idx[idx][value];
            } else {
                Word2Idx[idx][value] = Word2Idx[idx].size();
                db_table[line_count][idx] = Word2Idx[idx][value];
                Idx2Word[idx][Word2Idx[idx].size() - 1] = value;
            }
            line_count += 1;
        }
        for(int i = 0; i < attr_size; ++i) {
            category_size[i] = Word2Idx[i].size();
        }
        ifs.close();
    }
}

void read_testData(ifstream& ifs, string& test_name) {
    test_db_table.clear();
    ifs.open(test_name);
    if(ifs.is_open()) {
        string line;
        getline(ifs, line);
        int idx = 0;
        while(getline(ifs, line)) {
            test_db_table.push_back(vector<int>(db_table[0].size(), -1));
            int now = 0;
            int i = 0;
            for(int pos = 0; pos < line.size(); ++pos) {
                if(line[pos] == '\t') {
                    string value = line.substr(now, pos-now);
                    test_db_table[idx][i] = Word2Idx[i][value];
                    now = pos+1;
                    i += 1;
                }
            }
            string value = line.substr(now);
            value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
            value.erase(std::remove(value.begin(), value.end(), '\n'), value.end());
            test_db_table[idx][i] = Word2Idx[i][value];
            idx += 1;
        }
        ifs.close();
    }
}
void write_resultData(ofstream& ofs, string& result_name) {
    ofs.open(result_name);
    for(int i = 0; i < attr_size; ++i) {
        ofs.write(attribute_name[i].c_str(), attribute_name[i].length());
        if(i != attr_size-1){
            ofs.write("\t", 1);
        } else {
            ofs.write("\n", 1);
        }
    }
    for(int i = 0; i < test_db_table.size(); ++i) {
        for(int j = 0; j < test_db_table[i].size() - 1; ++j) {
            int label = test_db_table[i][j];
            ofs.write(Idx2Word[j][label].c_str(), Idx2Word[j][label].length());

            ofs.write("\t", 1);
        }
        string pred = Idx2Word[attr_size-1][result[i]];
        ofs.write(pred.c_str(), pred.length());
        ofs.write("\n", 1);
    }
    ofs.close();
}

int main(int argc, char* argv[]) {
    string train_name, test_name, result_name;
    ifstream ifs;
    ofstream ofs;
    if(argc != 4) {
        cout << "Usage: " << argv[0] << " train_name test_name result_name\n";
        return 1;
    }
    train_name = argv[1];
    test_name = argv[2];
    result_name = argv[3];
    read_trainData(ifs, train_name);
    /*
    decisionTree t[3];
    for(int type = 0; type < 3; ++type) {
        t[type] = makeDecisionTree(type);
    }
    */
    decisionTree t = makeDecisionTree(2);
    read_testData(ifs, test_name);
    for(int i = 0; i < test_db_table.size(); ++i) {
        vector<int> line = test_db_table[i];
        vector<int> count(category_size[attr_size-1], 0);
        /*
        int ret = -1;
        for(int type = 0; type < 3; ++type) {
            int tmp = getPredictedIdx(t[type], line, type);
            count[tmp] += 1;
            if(type == 2) ret = tmp;
        }
        for(int i = 0; i < count.size(); ++i) {
            if(count[i] >= 2) {
                ret = i;
            }
        }
        */
        int ret = getPredictedIdx(t, line, 2);

        result.push_back(ret);
    }
    
    write_resultData(ofs, result_name);
}
