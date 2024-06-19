#include "VNAME.h"
#include "verilated.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

static unsigned long next_rand = 1;
/* Suggested by the rand(3) manpage */
int myrand()
{
  next_rand = next_rand * 1103515245l + 12345l;
  return next_rand >> 16 & 0x7fffffff;
}

void getInputVector(std::string str, std::vector<int> &vec) {
  size_t last = 0;
  size_t next = 0;
  
  while ((next = str.find(",", last)) != std::string::npos) {
    std::string s = str.substr(last, next-last);
    if (s == "n") {
      vec.push_back(-1);
    } else {
      vec.push_back(stoi(s));
    }
    last = next + 1;
  }
  std::string s = str.substr(last);
  if (s == "n") {
    vec.push_back(-1);
  } else {
    vec.push_back(stoi(s));
  }
}

bool acceptInput(CData ready, CData valid, QData data, std::vector<int> &offered, std::vector<int> &accepted) {
  if (ready && valid) {
    accepted.push_back(data);
    offered.pop_back();
    return true;
  }
  return false;
}

void recordOutput(CData ready, CData valid, QData data, std::vector<int> &output) {
  if (ready && valid) {
    output.push_back(data);
  }
}

/*
@ ready & valid
   ? ready
   ! valid
   " "  neither ready nor valid
*/
void trace(std::ofstream &traceFile, CData ready, CData valid, QData data) {
  traceFile << (ready ? (valid ? '@' : '?') : (valid ? '!' : ' '));
  if (ready && valid) {
    traceFile << std::setw(2) << data << " ";
  } else {
    traceFile << "-- ";
  }
}

void printResult(std::ofstream &outFile, std::vector<int> &vec) {
  for (auto i : vec)
    outFile << i << " ";
  outFile << std::endl;
}

int main(int argc, char **argv) {

  Verilated::commandArgs(argc, argv);
  auto *tb = new VNAME;

  // SETUP
  
  vluint64_t main_time = 0;

  std::ofstream traceFile;
  std::stringstream ss;
  ss << argv[1] << ".trace";
  traceFile.open(ss.str());
  
  std::ofstream outFile;
  ss.str("");
  ss << argv[1] << ".out";
  outFile.open(ss.str());

  int delay = 2000*4;

  // reset
  tb->reset = 1;
  tb->clock = 0;
  tb->eval();
  tb->clock = 1;
  tb->eval();
  tb->reset = 0;

  while (EXIST (delay > 0)) { 
    switch (main_time & 0x3) {
    case 0: tb->clock = 1; break;

    case 1:
      // HANDSHAKE

      break;
      
    case 2:
      tb->clock = 0;

      // INPUTS

      break;

    case 3:
      traceFile << std::dec << std::setw(5) << (main_time/4) << ' ' << std::hex << std::setw(2);
      // TRACE

      traceFile << std::endl;
      
      break;
    }
    
    tb->eval();
    main_time++;
    DELAY
  }

  outFile << std::hex;
  // RESULTS

  traceFile.close();
  outFile.close();
  exit(EXIT_SUCCESS);
}
