#pragma once

#include <queue>
#include <mutex>
#include <thread>

template <typename T, typename Greater = std::greater<T>>
class FixedMinPriorityQueue : public std::priority_queue<T, std::vector<T>, Greater>
{
public:
	typedef std::priority_queue<T, std::vector<T>, Greater> PQueue;
	FixedMinPriorityQueue(const size_t max_size, Greater greater = Greater())
		: PQueue(greater), max_size_(max_size), greater_(greater) {}

	bool
	push(const T& x)
	{
		const std::lock_guard<std::mutex> lock(mutex_);
		if (PQueue::size() == max_size_)
		{
			if (greater_(x, PQueue::top()))
			{
				PQueue::pop();
			}
			else
			{
				return false;
			}
		}
		PQueue::push(x);
		return true;
	}

	const std::vector<T>&
	data(){
		return PQueue::c;
	}
private:
	std::mutex mutex_;
	const size_t max_size_;
	Greater greater_;
};