# DPL-en-handgeschreven-sudokus
Dit is de sourcecode gebruikt voor het uitvoeren van de testen in de paper geschreven door Brent Van Calsteren en Lennert Nijs:
Neurosymbolisch systeem voor het oplossen van handgeschreven sudoku’s
Hier hebben we Deeproblog gebruikt voor het trainen van een neuraal net dat handgeschreven cijfers kan herkennen.
Er wordt voor het trainen enkel een sudoku bestaande uit handgeschreven cijfers meegeven alsook of deze oplosbaar is.
Het project bevat alle nodige code van DeepProblog al (geschreven door Robin Manheave).
Extra librabries voor het gebruik van de code moeten wel nog geinstaleerd worden, check hiervoor DPL git: https://github.com/ML-KULeuven/deepproblog

## sudoku generator
Verder voor het trainen moesten we ook in staat zijn snel sudoku's te kunnen trainen, we hebben hiervoor beroep gedaan op volgende code van Bart Van den Broeck, Piet Goris: Het genereren van moeilijke Sudokus, https://gitlab.com/vdbroeckb/sudokugenerator 
