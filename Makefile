JC = javac
JFLAGS =
SRCS = ${wildcard *.java}
# SRCS = Main.java
OBJS = ${SRCS:.java=.class}

.SUFFIXES: .java .class

.java.class:
	$(JC) $(JFLAGS) $*.java

all: $(OBJS)

# modify the zip command so it's appropriate for your project
submit:
	zip submit.zip $(SRCS) Makefile LICENSE  HONOR ChatGPT_Transcript monks1.te.dta monks1.tr.dta monks2.te.dta monks2.tr.dta monks3.te.dta monks3.tr.dta mushroom.dta sonar.dta votes.dta xor.dta iris-binary.dta

clean:
	rm -f *.class

