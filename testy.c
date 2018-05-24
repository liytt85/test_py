#include <iostream>
#include <vector>


using namespace std;
struct ListNode
{
	int val;
	struct ListNode *next;
	ListNode(int x){
		val = x;
		next = NULL;

		}
		
	
};
int main(){
	struct ListNode head(8);
	ListNode nextone(3);
	head.next = &nextone;
	cout << head.val << (*head.next).val << endl;
	vector
}
