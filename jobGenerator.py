#!/usr/bin/env python3

import click
import sys
import numpy as np
import scipy.stats as dists
import matplotlib.pyplot as plt
import random

choiceList = ["cyclic", "all", "tree"]

class jobGenerator():
  def __init__(self, numjobs, gpus):
    self.numjobs = int(numjobs)
    self.gpus = int(gpus)

  def generateJobs(self):
    gpuList = self.randomRange(1, self.gpus)
    execTimeList = self.randomRange(1, 25)
    patternList = self.randomChoice(choiceList)
    bwSensitiveList = self.randomRange(0, 2)
    return zip(gpuList, patternList, execTimeList, bwSensitiveList)

  def randomRange(self, min, max):
    return [str(random.randrange(min, max)) for _ in range(self.numjobs)]
  
  def randomChoice(self, choiceList):
    return [str(random.choice(choiceList)) for _ in range(self.numjobs)]


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
