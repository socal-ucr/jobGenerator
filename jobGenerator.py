#!/usr/bin/env python3

import click
import sys
import random

choiceList = ["cyclic", "all", "tree"]

class jobGenerator():
  def __init__(self, numjobs, gpus):
    self.numjobs = int(numjobs)
    self.gpus = int(gpus)

  def generateJobs(self):
    gpuList = self.randomRange(1, self.gpus)
    arvlTimeList = self.randomProgression(50)
    srvcTimeList = self.randomRange(5, 25)
    patternList = self.randomChoice(choiceList)
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
@click.option('--numgpus', default=8, help='Number of GPUs')
@click.option('--outfile', prompt='OutFile', help='OutFile.txt')
def main(numjobs, numgpus, outfile):
  gen = jobGenerator(numjobs, numgpus)
  jobs = gen.generateJobs()
  f = open(outfile, "w")
  for job in jobs:
    toWrite = ",".join(job)
    f.write(toWrite + "\n")
  f.close()


if __name__ == "__main__":
  main()
