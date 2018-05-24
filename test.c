#include <iostream>
#include <stdlib.h>
#include <curses.h>

using namespace std;
void init_board(int (&board)[3][3]){
	srand(9);
	int row = rand() % 3 ;
	int clom = rand() % 3;
	for (int num=8; num>=1; num--){
		if (row == 2 && clom == 2){
			row = 0;
			clom = 0;
		}
		board[row][clom] = num;
		clom += 1;
		if (clom == 3){
			row += 1;
			clom = 0;
		}
	}
	board[2][2] = 0;
	for (int out_row = 0; out_row <= 2; out_row++){
		cout << board[out_row][0] << "  " << board[out_row][1] << "  " << board[out_row][2] << endl;
	}
}

//receive the input of keyboard
int ret_answer( int solu){
	do {
		int front = getch();
		int key_input = getch();
		switch(key_input){
			case 72:
				solu = 2;//top
				break;
			case 75:
				solu = 1;//left
				break;
			case 77:
				solu = 3;//right
				break;
			case 80:
				solu = 4;//down
				break;
			default:
				break;
		}

	}while(solu == 0);
	return solu;

}
void out_answer(int *row_1, int *clom_1, int solu, int (&board)[3][3]){
	int row = *row_1;
	int clom = *clom_1;
	if  (solu == 2){
		//top
		if (row == 2){
			;
		}
		else{
			board[row][clom] = board[row+1][clom];
			board[row+1][clom] = 0;
			*row_1 += 1;
		}
	}
	if  (solu == 1){
		//left
		if (clom == 0){
			;
		}
		else{
			board[row][clom] = board[row][clom-1];
			board[row][clom] = 0;
			*clom_1 -= 0;
		}
	}
	if  (solu == 3){
		//right
		if  (clom == 2){
			;
		}
		else{
			board[row][clom] = board[row][clom+1];
			board[row][clom+1] = 0;
			*clom_1 += 1;
		}
	}
	if (solu == 4){
		//down
		if  (row == 0){
			;
		}
		else{
			board[row][clom] = board[row-1][clom];
			board[row-1][clom] = 0;
			*row_1 -= 1;
		}
	}
	for (int out_row = 0; out_row <= 2; out_row++){
		cout << board[out_row][0] << "  " << board[out_row][1] << "  " << board[out_row][2] << endl;
	}
}
int main(){
	int keyboard_input = 0;
	int ret_input;
	int result = getch();
	cout << result << endl;
	int row_1[3][3] = {1} , row_2[3][3] = {0} , row_3[3][3] = {0};
	init_board(row_1);
	int row = 0;
	int clom = 0;
	while(1){
		ret_input = ret_answer(keyboard_input);
		out_answer(&row,&clom,ret_input,row_1);
	}
	cout << row_2[1][0]  << endl;
	return 0;
}
