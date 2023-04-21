nn(mnist_net,[X],Y,[blah,1,2,3,4]) :: convertFoto(X,Y).

%Sudoku is represented as a 2d list
checkValidSudoku(In):-
    basicChecks(In,Lenght,Root,Li),
    linkWithVars(In,Links-Vars),
    generate3Lists2D(Vars,Lenght,Root,L1-L2-L3),
    getForEachElAllRows(Vars,L1-L2-L3,Vars-List),
    relinkNumbers(Vars,Links),
    checkContraintEachEl(Vars-List,Links,Li).
%-----------------------------------------------
relinkNumbers([],[]).
relinkNumbers([A|Ax],[B|Bx]):-
     relinkNumber(A,B), relinkNumbers(Ax,Bx).

relinkNumber([],[]).
relinkNumber([V|Var],[L|Link]):-
    number_custom(L), V = L,relinkNumber(Var,Link).
relinkNumber([_V|Var],[L|Link]):-
    compound_custom(L),relinkNumber(Var,Link).
relinkNumber([_V|Var],[L|Link]):-
    var_custom(L),relinkNumber(Var,Link).

linkWithVars([],[]-[]).
linkWithVars([El|L],[Link|Links]-[Var|Vars]):-
    linkWithVar(El,Link-Var),linkWithVars(L,Links-Vars).

linkWithVar([],[]-[]).
linkWithVar([El|L],[El|Links]-[_|Vars]):- compound_custom(El),linkWithVar(L,Links-Vars).
linkWithVar([El|L],[El|Links]-[_|Vars]):- number_custom(El),linkWithVar(L,Links-Vars).
linkWithVar([empty|L],[A|Links]-[A|Vars]):-linkWithVar(L,Links-Vars).
%linkWithVar([El|L],[A|Links]-[A|Vars]):-lowEquals(El,"empty"),linkWithVar(L,Links-Vars).
linkWithVar([El|L],[El|Links]-[El|Vars]):-var_custom(El),linkWithVar(L,Links-Vars).

%-----------------------------------------------

%-----------------------------------------------

checkContraintEachEl([]-[],[],_).
checkContraintEachEl([E|El]-[Li|Lis],[Link|Links],List):-
    checkContraintEl(E-Li,Link,List),
    checkContraintEachEl(El-Lis,Links,List).

checkContraintEl([]-[],[],_).
checkContraintEl([E|El]-[Li|List],[Link|Links],L):-
    checkContraint(E,Link,Li,L),checkContraintEl(El-List,Links,L).

checkContraint(E,Link,L,List):- var_custom(Link),member(E,List), nomemberCostum(E,L).
checkContraint(E,Link,L,List):-
    compound_custom(Link), convertFoto(Link,E),nomemberCostum(E,L).
checkContraint(E,_Link,_L,_):-number_custom(E).

%-----------------------------------------------
%-----------------------------------------------


getForEachElAllRows([],_,[]-[]).
getForEachElAllRows([El|In],L1-L2-L3,[El|List]-[Rows|Rest]):-
    getEachElAllRows(El,L1-L2-L3,El-Rows),
    getForEachElAllRows(In,L1-L2-L3,List-Rest).


getEachElAllRows([],_,[]-[]).
getEachElAllRows([El|In],L1-L2-L3,[El|List]-[Rows|Rest]):-
    getElAllRows(El,L1-L2-L3,Rows),
    getEachElAllRows(In,L1-L2-L3,List-Rest).


getElAllRows(El,L1-L2-L3,R5):-
    getRow(El,L1,R1),
    getRow(El,L2,R2),
    getRow(El,L3,R3),
    unionList(R1,R2,R4),
    unionList(R3,R4,R5).

getRow(_,[],_):-false.
getRow(El,[Row|L],R):-nomemberCostum(El,Row),getRow(El,L,R).
getRow(El,[Row|_],Rest):-memberCostum(El,Row),removeCostum(Row,El,Rest).


%unifies 2 lists
unionList([],L,L).
unionList(L,[],L).
unionList([H|T],L2,L):-
    memberCostum(H,L2),
    unionList(T,L2,L).
unionList([H|T],L2,[H|L]):-
    nomemberCostum(H,L2),
    unionList(T,L2,L).
%-----------------------------------------------
%-----------------------------------------------


%the easy checks
basicChecks(Sudoku,L,Root,Li):-
    L = 9,
    Root = 3,
    list_length(Sudoku,L),
    checkLength(Sudoku,L),
    Li = [1,2,3,4,5,6,7,8,9].
 basicChecks(Sudoku,L,Root,Li):-
    L = 4,
    Root = 2,
    list_length(Sudoku,L),
    checkLength(Sudoku,L),
    Li = [1,2,3,4].

%checks length of list
checkLength([],_L).
checkLength([El|List],L):- list_length(El,L),checkLength(List,L).

%-----------------------------------------------
%-----------------------------------------------


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
    append_custom(Buffer,Sub,App),
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
    append_custom(Buffer,SubL,App),
    knot1DList(Start,End,Xs,App,Elist).

% Returns a sublist containing elements between index start-end (start
% included, end not)
getSubL(_,_,[],[]).
getSubL(Start,End,List,SubL):- getSubLB(Start,End,List,[],SubL).
getSubLB(Start,End,_List,SubL,SubL):-
    Start >= End.
getSubLB(Start,End,List,Buffer,SubL):-
    Start < End, Inc is Start + 1,
    nth0_custom(Start, List, R),
    getSubLB(Inc,End,List,[R|Buffer],SubL).
%------------------------------------------------------------

%buildin replacements
removeCostum([],_,[]).
removeCostum([X|List],Var,List):- X == Var.
removeCostum([X|List],Var,[X|Rest]):-X \== Var,removeCostum(List,Var,Rest).


% had 2 write this since buildin member
% didn't work with variables as I wanted
memberCostum(_,[]):- false.
memberCostum(Var,[X|_Xs]):- X == Var.
memberCostum(Var,[_X|Xs]):- memberCostum(Var,Xs).

nomemberCostum(_,[]).
nomemberCostum(Var,[X|Xs]):-X \== Var, nomemberCostum(Var,Xs).


list_length([], 0).
list_length([_|T], Len) :-
    list_length(T, LenT),
    Len is LenT + 1.

nth0_custom(N, List, Elem) :-
    nth0_custom(N, List, 0, Elem).

nth0_custom(N, [H|_], N, H).
nth0_custom(N, [_|T], I, Elem) :-
    I < N,
    I1 is I + 1,
    nth0_custom(N, T, I1, Elem).

append_custom([], List, List).
append_custom([H|T], List, [H|Result]) :-
    append_custom(T, List, Result).

var_custom(X) :- var(X).
number_custom(X) :-number(X).

compound_custom(X) :-compound(X).

%lowEquals(X,Y):- X == Y.

