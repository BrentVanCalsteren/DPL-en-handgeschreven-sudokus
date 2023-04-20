:- use_module(library(apply)).
nn(mnist_net,[X],Y,[1,2,3,4]) :: convertFoto(X,Y).

%Sudoku is represented as a 2d list
checkValidSudoku(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P, Out):-
    In = [[A,B,C,D],[E,F,G,H],[I,J,K,L],[M,N,O,P]],
    basicChecks(In,Lenght,Root,Li),
    replaceEmptyWithVars(In,Temp,Vtemp-Value),
    generate3Lists2D(Temp,Lenght,Root,L1-L2-L3),
    getForEachElAllRows(Temp,L1-L2-L3,ElT-ListT),
    replaceBackNonTemps(ElT-ListT,Vtemp-Value,El-List),
    hasAllUniqueNumberss(List),
    checkContraintEachEl(El-List,Li,Result),
    Result = Out.


%unifies 2 lists
unionList([],L,L).
unionList(L,[],L).
unionList([H|T],L2,[H|L]):-
    not(memberCostum(H,L2)),
    unionList(T,L2,L).
unionList([H|T],L2,L):-
    memberCostum(H,L2),
    unionList(T,L2,L).




checkContraintEachEl([]-[],_,[]).
checkContraintEachEl([E|El]-[Li|Lis],List,[R|Result]):-
    checkContraintEl(E-Li,List,R),
    checkContraintEachEl(El-Lis,List,Result).


checkContraintEl([]-[],_,[]).
checkContraintEl([E|El]-[Li|List],L,[R|Result]):-
    checkContraint(E,Li,L,R),checkContraintEl(El-List,L,Result).

checkContraint(E,L,List,E):-
    not(compound(E)),var(E),member(E,List),
    not(memberCostum(E,L)).

checkContraint(E,L,List,C):-
    compound(E),
    member(C,List),
    not(memberCostum(C,L)),
    convertFoto(E,C).

checkContraint(E,_,_,E):-not(compound(E)),not(var(E)).


replaceBackNonTemps(ElT-ListT,Vtemp-Value,El-List):-
    replaceElss(ElT,Vtemp-Value,El),
    replaceListss(ListT,Vtemp-Value,List).

replaceListss([],_,[]).
replaceListss([L1|ListT],Vtemp-Value,[R1|Result]):-
    replaceLists(L1,Vtemp-Value,R1),
    replaceListss(ListT,Vtemp-Value,Result).


replaceLists([],_,[]).
replaceLists([L|ListT],Vtemp-Value,[R|Result]):-
    replaceEls(L,Vtemp-Value,R),
    replaceLists(ListT,Vtemp-Value,Result).


replaceElss([],_,[]).
replaceElss([E|ElT],Vtemp-Value,[R|Result]):-
    replaceEls(E,Vtemp-Value,R),replaceElss(ElT,Vtemp-Value,Result).

replaceEls([],_,[]).
replaceEls([E|ElT],Vtemp-Value,[R|Result]):-
    replaceEl(E,Vtemp-Value,R),replaceEls(ElT,Vtemp-Value,Result).


replaceEl(_,[]-[],_).
replaceEl(E,[Ve|Vtemp]-[Va|Value],R):-
    not(replaceel(E,Ve-Va,R)),replaceEl(E,Vtemp-Value,R).
replaceEl(E,[Ve|_]-[Va|_],R):- replaceel(E,Ve-Va,R).


replaceel(E,[]-[],E):-not(compound(E)).
replaceel(E,[Ve|Vtemp]-[_Va|Value],R):-not(E==Ve), replaceel(E,Vtemp-Value,R).
replaceel(E,[Ve|_Vtemp]-[Va|_Value],Va):- E==Ve.



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
getRow(El,[Row|L],R):-not(memberCostum(El,Row)),getRow(El,L,R).
getRow(El,[Row|_],Rest):-memberCostum(El,Row),removeCostum(Row,El,Rest).



removeCostum([],_,[]).
removeCostum([X|List],Var,List):- X == Var.
    %(not(var(X)),not(compound(X)),not(var(Var)),not(compound(Var)),Var = X).
removeCostum([X|List],Var,[X|Rest]):-not(Var == X),removeCostum(List,Var,Rest).



% had 2 write this since buildin member
% didn't work with variables as I wanted
memberCostum(_,[]):- false.
memberCostum(Var,[X|Xs]):- not(X == Var),memberCostum(Var,Xs).
memberCostum(Var,[X|_Xs]):- X == Var.



replaceEmptyWithVars([],[],[]-[]).
replaceEmptyWithVars([El|L],[O|Out],[Temp|RTemp]-[Value|RValue]):-
    replaceEmptyWithVar(El,O,Temp-Value),replaceEmptyWithVars(L,Out,RTemp-RValue).


replaceEmptyWithVar([],[],[]-[]).
replaceEmptyWithVar([El|L],[temp(A)|Out],[temp(A)|Temps]-[El|Values]):- not(El == empty), replaceEmptyWithVar(L,Out,Temps-Values).
replaceEmptyWithVar([El|L],[_|Out],Rest):- El == empty, replaceEmptyWithVar(L,Out,Rest).

%the easy checks
basicChecks(Sudoku,L,Root,Li):-
    length(Sudoku,L),
    findall(Num, between(1, L, Num), Li),
    maplist(checkLength(L),Sudoku),
    Sqr is sqrt(L),
    Root is floor(Sqr),
    Root =:= Sqr.

%checks length of list
checkLength(L,List):- length(List,L).

hasAllUniqueNumberss([]).
hasAllUniqueNumberss([I|In]):-
    hasAllUniqueNumbers(I),hasAllUniqueNumberss(In).

%checks if all the contraints are met
hasAllUniqueNumbers([]).
hasAllUniqueNumbers([N1|Rest]):-
   check(N1,N1),hasAllUniqueNumbers(Rest).

check([],_).
check(_Check,[]):- false.
check([N|Check],L2):-  not(N == empty),memberCostum(N,L2),removeCostum(L2,N,Rest),check(Check,Rest).
check([N|Check],L2):-  N == empty, check(Check,L2).
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

