#!/usr/bin/env python3

import click
import sys
import random

choiceList = ["ring", "all", "tree"]

class jobGenerator():
  def __init__(self, numjobs, mingpus, maxgpus, patternList):
    self.numjobs = int(numjobs)
    self.mingpus = int(mingpus)
    self.maxgpus = int(maxgpus)
    self.patternList = patternList

  def generateJobs(self):
    gpuList = self.randomRange(self.mingpus, self.maxgpus)
    arvlTimeList = self.randomProgression(50)
    srvcTimeList = self.randomRange(5, 25)
    patternList = self.randomChoice(self.patternList)
    bwSensitiveList = self.randomRange(0, 2)
    return zip(gpuList, patternList, arvlTimeList, srvcTimeList, bwSensitiveList)

  def randomRange(self, minimum, maximum, dataType=str):
    return [dataType(random.randrange(minimum, maximum)) for _ in range(self.numjobs)]
  
  def randomChoice(self, choiceList):
    return [str(random.choice(choiceList)) for _ in range(self.numjobs)]

  def randomProgression(self, maximum):
    samples = self.randomRange(0, maximum, int)
    return [str(i) for i in sorted(samples)]


@click.command()
@click.option('--numjobs', default=1000, help='Number of Jobs')
@click.option('--mingpus', default=2, help='Number of GPUs')
@click.option('--maxgpus', default=8, help='Number of GPUs')
@click.option('--pattern', default="random", help='Pattern of AppGraph')
@click.option('--outfile', prompt='OutFile', help='OutFile.txt')
def main(numjobs, mingpus, maxgpus, pattern, outfile):
  patternList = []
  if (str(pattern) == "random"):
    patternList = choiceList
  else:
    patternList = [str(pattern)]
  gen = jobGenerator(numjobs, mingpus, maxgpus, patternList)
  jobs = gen.generateJobs()
  f = open(outfile, "w")
  for job in jobs:
    toWrite = ",".join(job)
    f.write(toWrite + "\n")
  f.close()


if __name__ == "__main__":
  main()
