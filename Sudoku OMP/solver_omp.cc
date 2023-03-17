#include <cstring>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <omp.h>
#include <iostream>
//#include <ittnotify.h>

#define MAX_PARALLEL_DEPTH 1

constexpr uint32_t kAll = 0x1ff; // 0b00000|000000000|000000000|111111111

struct SolverImpl {
  std::vector<uint32_t> cells_todo;
  size_t num_todo = 0; 
  bool num_solutions = 0;

  bool Solve(size_t todo_index, char *solution, std::array<uint32_t, 9> &rows, std::array<uint32_t, 9> &cols, std::array<uint32_t, 9> &boxes) { // recursive algorithm
    if(num_solutions!=0) {return false;}

    uint32_t row_col_box = cells_todo[todo_index];
    auto row = row_col_box & kAll; // masking. Get first (leftmost) 9 bits.1 bits represent possible values, 0 bits are invalid values to use.
    auto col = (row_col_box >> 9) & kAll; // gets next 9 bits and masks them
    auto box = (row_col_box >> 18) & kAll; // gets next 9 bits again and mask
    auto candidates = rows[row] & cols[col] & boxes[box]; // condidates is last 9 bits which are 0 or 1. Index correxponding to 1 is a valid possible solution
    bool solved = false; 

    if (todo_index >= num_todo) { // solved last empty square
      if (num_solutions>0) {
        return false;
      }
      bool local_success = false; 
      #pragma omp critical  
      {
        local_success = num_solutions==0; // only if num_solutions==0, let local_success=1
        num_solutions=1;
      } 

      if(local_success) {
        uint32_t candidate = candidates & -candidates; // i & (~i + 1). Trick to extract lowest set bit
        solution[row * 9 + col] = (char)('0' + __builtin_ffs(candidate));
      }

      return local_success;
    }

    while (candidates && num_solutions==0) { // While a possible solution exits  
      uint32_t candidate = candidates & -candidates; // i & (~i + 1). Trick to extract lowest set bit
      candidates = candidates & (candidates - 1); // remove tried (but invalid) solution


      //===== recursion section =====
        #pragma omp task firstprivate(rows, cols, boxes) shared(solved) priority(todo_index)
        {
          rows[row] ^= candidate; // invalidate chosen number from set of valid numbers
          cols[col] ^= candidate; 
          boxes[box] ^= candidate;
          int result;

          if (todo_index <= MAX_PARALLEL_DEPTH)
            result = Solve(todo_index + 1, solution, rows, cols, boxes); // continue to solve next empty square recursively
          else
            result = Solve_Sequential(todo_index + 1, solution, &rows, &cols, &boxes);
          
          if (result) {
            solved = true;
            solution[row * 9 + col] = (char)('0' + __builtin_ffs(candidate)); // get index of chosen candidate (our guess) and turn it into a char
          }
        } // end pragma task
        if (solved){
          return solved;
        }
      //=====

      //rows[row] ^= candidate; // undo invalidation
      //cols[col] ^= candidate;
      //boxes[box] ^= candidate;
    } // END WHILE

    if (!solved) {
      #pragma omp taskwait
    }
    
    return solved;
  }

  bool Solve_Sequential(size_t todo_index, char *solution, std::array<uint32_t, 9>* rows, std::array<uint32_t, 9>* cols, std::array<uint32_t, 9>* boxes) {
    if (num_solutions!=0) { // already solved by another process
      return false;
    }

    uint32_t row_col_box = cells_todo[todo_index];
    auto row = row_col_box & kAll; // masking. Get first (leftmost) 9 bits.1 bits represent possible values, 0 bits are invalid values to use.
    auto col = (row_col_box >> 9) & kAll; // gets next 9 bits and masks them
    auto box = (row_col_box >> 18) & kAll; // gets next 9 bits again and mask
    auto candidates = (*rows)[row] & (*cols)[col] & (*boxes)[box]; // condidates is last 9 bits which are 0 or 1. Index correxponding to 1 is a valid possible solution
    
    if (todo_index >= num_todo) { // solved last empty square
      bool local_success = true; // no atomic XNOR, so set this to true
      #pragma omp atomic 
      local_success = local_success ^ num_solutions; // only if num_solutions==0, let local_success=1
      
      num_solutions=1;
      if(local_success){
        uint32_t candidate = candidates & -candidates; // i & (~i + 1). Trick to extract lowest set bit 
        solution[row * 9 + col] = (char)('0' + __builtin_ffs(candidate));
      }
      return local_success;
    }
    
    while (candidates && num_solutions==0) { // While a possible soln exits
      uint32_t candidate = candidates & -candidates; // i & (~i + 1). Trick to extract lowest set bit 
      (*rows)[row] ^= candidate; // invalidate chosen number from set of valid numbers
      (*cols)[col] ^= candidate;
      (*boxes)[box] ^= candidate;

      //if (todo_index < num_todo) {
        int result = Solve_Sequential(todo_index + 1, solution, rows, cols, boxes); // continue to solve next empty square recursively
        if (result) { // if solution found, do not search for more
          solution[row * 9 + col] = (char)('0' + __builtin_ffs(candidate));
          return true;
        }
      
      
      (*rows)[row] ^= candidate; // undo invalidation
      (*cols)[col] ^= candidate;
      (*boxes)[box] ^= candidate;
      candidates = candidates & (candidates - 1);
    } // end while
    return false; // no solution found
  }

  bool Initialize(const char *input, char *solution, std::array<uint32_t, 9>& rows, std::array<uint32_t, 9>& cols, std::array<uint32_t, 9>& boxes) {
    rows.fill(kAll);
    cols.fill(kAll);
    boxes.fill(kAll);
    cells_todo.clear();
    num_solutions = 0;
    memcpy(solution, input, 81);

    for (int row = 0; row < 9; ++row) {
      for (int col = 0; col < 9; ++col) {
        int box = int(row / 3) * 3 + int(col / 3);
        if (input[row * 9 + col] == '.') {
          cells_todo.emplace_back(row | (col << 9) | (box << 18)); // encode location of todo cell into uint32
        } else { // check if out given input is valid
          uint32_t value = 1u << (uint32_t)(input[row * 9 + col] - '1'); // encode number in sudoku to 9 bitmap
          if (rows[row] & value && cols[col] & value && boxes[box] & value) { // check validity of row, col & box
            rows[row] ^= value; // update row, col & box, s.t. index number will be set as 0
            cols[col] ^= value; // ^ is XOR bitwise operation
            boxes[box] ^= value;
          } else { // invalid input given
            return false;
          }
        }
      }
    }
    num_todo = cells_todo.size() - 1; // last index of cell to be solved???
    return true;
  }
};


extern "C" size_t Solver(const char *input, char *solution) {
  //__itt_domain* domain = __itt_domain_create("Example.Domain.Global");
  //__itt_string_handle* handle_main = __itt_string_handle_create("main");
  static SolverImpl solver;
  std::array<uint32_t, 9> rows{}, cols{}, boxes{};
  int var1 = 10;
  if (solver.Initialize(input, solution, rows, cols, boxes)) {
    #pragma omp parallel
      #pragma omp single
        solver.Solve(0, solution, rows, cols, boxes);
    return solver.num_solutions;
  }
  return 0;
}

