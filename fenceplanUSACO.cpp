#include <string>
#include <fstream>
#include <array>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <queue>
using namespace std;

// link to USACO Silver 2019 US Open Contest Problem 3: Fence Planning
// https://usaco.org/index.php?page=viewproblem2&cpid=944

vector<int> connected[100001];
vector<pair<int, int> > cowLocs;
vector<bool> visited;
int numCows, numPairs;
int minX = 1000000000, maxX = 0, minY = 1000000000, maxY = 0;

// conducts depth-first search
int dfs(int node) {
    visited[node] = true;
    minX = min(minX, cowLocs[node - 1].first);
    maxX = max(maxX, cowLocs[node - 1].first);
    minY = min(minY, cowLocs[node - 1].second);
    maxY = max(maxY, cowLocs[node - 1].second);

    int c = 0;
    
    for(int i : connected[node])
    {
    	if(visited[i])
    		continue;
    	visited[i] = true;
    	c += dfs(i) + 1;
    }
    return c;
}

 
int main()
{

	ifstream fin("fenceplan.in");
	ofstream fout("fenceplan.out");

	fin >> numCows >> numPairs;
	
	for(int i = 0; i < numCows; i++)
	{
		int x, y;
		fin >> x >> y;
		cowLocs.push_back(make_pair(x, y));
	}

	for(int i = 0; i < numPairs; i++)
	{
		int cow1, cow2;
		fin >> cow1 >> cow2;
		connected[cow1].push_back(cow2);
		connected[cow2].push_back(cow1);
	}

	int ans = 1000000000;
	visited.assign(numCows + 1, false);
	for(int i = 1; i <= numCows; i++)
	{
		if(!visited[i])
		{
			minX = 1000000000, maxX = 0, minY = 1000000000, maxY = 0;
			int cowsReached = dfs(i) + 1;
			int perim = (maxX - minX) + (maxY - minY);
			ans = min(ans, 2 * perim);
		}
	}
	fout << ans << endl;

}