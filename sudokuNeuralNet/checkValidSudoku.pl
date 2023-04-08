:- use_module(library(apply)).
nn(mnist_net,[X],Y,[1,2,3,4]) :: convertFoto(X,Y).


%PS: THIS CODE IS FOR CHEKCING SUDOKUS THE ARE COMPLETED

test4x4([[_,_,_,_],[_,_,_,_],[_,_,_,_],[_,_,_,_]]).
test9x9([[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_],[_,_,_,_,_,_,_,_,_]]).


%Sudoku is represented as a 2d list
%Sudoku is represented as a 2d list
checkValidSudoku(In, Out):-
    print(In),
    convertDigits(In,Out,Con),
    basicChecks(Con,L,Root),
    harderChecks(Con,L,Root).

convertDigits([],[],[]).
convertDigits([L|Sudoku],[O|Out],[C|Con]):-
    convertDigit(L,O,C), convertDigits(Sudoku, Out,Con).

convertDigit([],[],[]).
convertDigit([El|L],[_O|Out],[O|Con]):- El \= empty, convertFoto(El,O), convertDigit(L,Out,Con).
convertDigit([empty|L],[_|Out],[_|Con]):- convertDigit(L,Out,Con).

%the easy checks
basicChecks(Sudoku,L,Root):-
    length(Sudoku,L),
    maplist(checkLength(L),Sudoku),
    Sqr is sqrt(L),
    Root is floor(Sqr),
    Root =:= Sqr.

%checks length of list
checkLength(L,List):- length(List,L).

%the harder checks
harderChecks(Sudoku,L,Rood):-
    generate3Lists2D(Sudoku,L,Rood,L1-L2-L3),
    findall(Num, between(1, L, Num), List),
    hasAllUniqueNumbers(L1,List),
    hasAllUniqueNumbers(L2,List),
    hasAllUniqueNumbers(L3,List).

%checks if all the contraints are met
hasAllUniqueNumbers([],_).
hasAllUniqueNumbers([],[]).
hasAllUniqueNumbers(_Check,[]):- false.
hasAllUniqueNumbers([N|Rest],L2):-
   check(N,L2) ,hasAllUniqueNumbers(Rest,L2).

check([],[]).
check(_Check,[]):- false.
check([N|Check],L2):-
    member(N,L2),delete(L2,N,Rest), check(Check,Rest).
    %convertFoto(N,Result), member(Result,L2),delete(L2,Result,Rest), check(Check,Rest).

generate3Lists2D(Sudoku,L,Root,Sudoku-Trans-Subsquares):-
    transpose(Sudoku,Trans),
    getSubSquares(0,L,Root,Sudoku,[],Subsquares).



/*
 * --- Code Transposing Sudoku ---
 */
% old implementation from SWI-PROLOG transpose
transpose([], []).
transpose([F|Fs], Ts) :-
    transpose(F, [F|Fs], Ts).

transpose([], _, []).
transpose([_|Rs], Ms, [Ts|Tss]) :-
        lists_firsts_rests(Ms, Ts, Ms1),
        transpose(Rs, Ms1, Tss).

lists_firsts_rests([], [], []).
lists_firsts_rests([[F|Os]|Rest], [F|Fs], [Os|Oss]) :-
        lists_firsts_rests(Rest, Fs, Oss).

/*
 * --- Code deviding Sudoku in subsquares
 * --- Square represented as 1d list
 */
%----------------------------------------------


%generate all subsquares
getSubSquares(_,_,_,[],_,_).
getSubSquares(Start,L,_,_,Squares,Squares):-
    Start >= L.
getSubSquares(Start,L,Size,Sudoku,Buffer,Subs):-
    End is Start + Size,
    getSubSquare(Start,End,L,Sudoku, Sub),
    append(Buffer,Sub,App),
    getSubSquares(End,L,Size,Sudoku,App,Subs).


%Return sub2dList of list in form of square size L
getSubSquare(Start,End,L,Sudoku,Square) :-
    getSubL(Start,End,Sudoku,Subs),
    knot2DList(0,L,Subs,[],Square).

%knot 2d list start -> end
knot2DList(_,_,[],_,[]).
knot2DList(Start,L,_,Elist,Elist):-
    Start >= L.
knot2DList(Start,L,List,Buffer,Elist):-
    End is Start + round(sqrt(L)),
    knot1DList(Start,End,List,[],Knot),
    knot2DList(End,L,List,[Knot|Buffer],Elist).


%Keep only the elements in list between Start & end
knot1DList(_,_,[],Elist,Elist).
knot1DList(Start,End,[X|Xs],Buffer,Elist):-
    getSubL(Start,End,X,SubL),
    append(Buffer,SubL,App),
    knot1DList(Start,End,Xs,App,Elist).

% Returns a sublist containing elements between index start-end (start
% included, end not)
getSubL(_,_,[],[]).
getSubL(Start,End,List,SubL):- getSubLB(Start,End,List,[],SubL).
getSubLB(Start,End,_List,SubL,SubL):-
    Start >= End.
getSubLB(Start,End,List,Buffer,SubL):-
    Start < End, Inc is Start + 1,
    nth0(Start, List, R),
    getSubLB(Inc,End,List,[R|Buffer],SubL).
%------------------------------------------------------------

