#include <string>
#include <fstream>
#include <array>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
using namespace std;

// link to USACO Silver 2018 December Contest Problem 2: Convention II
// https://usaco.org/index.php?page=viewproblem2&cpid=859

typedef long long LL;
typedef pair<LL, LL> pll;

int main()
{
	ifstream fin("convention2.in");
	ofstream fout("convention2.out");

	int numCows;	fin >> numCows;
	vector<pair<int, pll> > cows;
	set<pll> waitingList;

	for(int i = 1; i <= numCows; i++)
	{
		int enterTime, eatingTime;
		fin >> enterTime >> eatingTime;
		cows.push_back(make_pair(enterTime, make_pair(i, eatingTime)));
	}

	sort(cows.begin(), cows.end());

	LL currFin = cows[0].first + cows[0].second.second;
	int nextCow = 1;
	LL ans = 0;

	while(nextCow < numCows || waitingList.size() > 0)
	{
		while(nextCow < numCows && cows[nextCow].first <= currFin)
		{
			waitingList.insert(make_pair(cows[nextCow].second.first, nextCow));
			nextCow++;
		}

		if(waitingList.size() == 0 && nextCow < numCows)
		{
			currFin = cows[nextCow].first + cows[nextCow].second.second;
			nextCow++;
		}
		else if(waitingList.size() > 0)
		{
			set<pll>::iterator senior = waitingList.begin();
			ans = max(ans, currFin - cows[senior->second].first);
			currFin += cows[senior->second].second.second;
			waitingList.erase(senior);
		}
	}

	fout << ans << endl;

}
