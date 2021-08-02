#!/usr/bin/env python3

import click
import sys
import random
from statistics import mean

# TODO(Kiran): Make choiceList = ["ring", "all", "tree"] when we want to randomize
choiceList = ["ring"]
bwSensitiveNets = ["alexnet.sh", "inception-v3.sh", "resnet-50.sh", "vgg16.sh"]
bwInsensitiveNets = ["googlenet.sh", "caffenet.sh"]

netList = ["alexnet.sh", "inception-v3.sh", "resnet-50.sh", "vgg16.sh",
           "googlenet.sh", "caffenet.sh"]

class jobGenerator():
  # Note(kiran): Default min and max srvcTime be changed?
  def __init__(self, numjobs, mingpus, maxgpus, patternList, minST=5, maxST=25):
    self.numjobs = int(numjobs)
    self.mingpus = int(mingpus)
    self.maxgpus = int(maxgpus)
    self.patternList = patternList
    self.minSrvcTime = minST
    self.maxSrvcTime = maxST

  def __randomRange(self, minimum, maximum, dataType=str):
    return [dataType(random.randrange(minimum, maximum)) for _ in range(self.numjobs)]

  def __randomChoice(self, choiceList):
    return [str(random.choice(choiceList)) for _ in range(self.numjobs)]

  def __randomProgression(self, maximum):
    samples = self.__randomRange(0, maximum, int)
    return [str(i) for i in sorted(samples)]

  def generateJobs(self):
    gpuList = []
    if self.mingpus == self.maxgpus:
      gpuList = [str(self.mingpus)] * self.numjobs
    else:
      gpuList = self.__randomRange(self.mingpus, self.maxgpus)
    # Note(kiran): Anything other than mean?
    arvlTimeList = self.__randomProgression(
        mean(range(self.minSrvcTime, self.maxSrvcTime)) * self.numjobs)
    srvcTimeList = self.__randomRange(self.minSrvcTime, self.maxSrvcTime)
    patternList = self.__randomChoice(self.patternList)
    bwSensitiveList = []
    taskList = self.__randomChoice(netList)
    for task in taskList:
      if task in bwInsensitiveNets:
        bwSensitiveList.append(str(0))
      else:
        bwSensitiveList.append(str(1))
    return zip(gpuList, patternList, arvlTimeList, srvcTimeList, bwSensitiveList, taskList)


@click.command()
@click.option('--numjobs', default=100, help='Number of Jobs')
@click.option('--mingpus', default=2, help='Number of GPUs')
@click.option('--maxgpus', default=6, help='Number of GPUs')
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
