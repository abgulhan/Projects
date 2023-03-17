#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
constexpr uint32_t kAll = 0x1ff; // 0b00000|000000000|000000000|111111111

struct SolverImpl {
  std::array<uint32_t, 9> rows{}, cols{}, boxes{};
  std::vector<uint32_t> cells_todo;
  size_t num_todo = 0, num_solutions = 0;

  void Solve(size_t todo_index, char *solution) { // recursive algorithm
    uint32_t row_col_box = cells_todo[todo_index];
    auto row = row_col_box & kAll; // masking. Get first (leftmost) 9 bits
    auto col = (row_col_box >> 9) & kAll; // gets next 9 bits and masks them
    auto box = (row_col_box >> 18) & kAll; // gets next 9 bits again and mask
    auto candidates = rows[row] & cols[col] & boxes[box]; // condidates is last 9 bits which are 0 or 1.
    while (candidates) { // While a possible soln exits   // Index correxponding to 1 is a valid possible solution
      uint32_t candidate = candidates & -candidates; // i & (~i + 1). Trick to extract lowest set bit 
      rows[row] ^= candidate; // invalidate chosen number from set of valid numbers
      cols[col] ^= candidate;
      boxes[box] ^= candidate;
      solution[row * 9 + col] = (char)('0' + __builtin_ffs(candidate));   // get index of chosen candidate (our guess)
      if (todo_index < num_todo) { // if did not solve last empty square  // and turn it into a char
        Solve(todo_index + 1, solution); // continue to solve next empty square recursively
      } else { // solved last empty square
        ++num_solutions; // record solution (make atomic in omp version)
      }
      if (num_solutions > 0) { // if any solution was found, return
        return;
      }
      rows[row] ^= candidate; // else, undo changes made by candidate (backtrack)
      cols[col] ^= candidate;
      boxes[box] ^= candidate;
      candidates = candidates & (candidates - 1);
    }
  }

  bool Initialize(const char *input, char *solution) {
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
          uint32_t value = 1u << (uint32_t)(input[row * 9 + col] - '1');
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
  static SolverImpl solver;
  if (solver.Initialize(input, solution)) {
    solver.Solve(0, solution);
    return solver.num_solutions;
  }
  return 0;
}
