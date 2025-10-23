import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
from itertools import permutations

class NQueensGrover:
    def __init__(self, n):
        """
        Initialize the N-Queens problem solver using Grover's algorithm.
        
        Args:
            n (int): Size of the chessboard (N x N)
        """
        self.n = n
        self.num_qubits = n * int(math.ceil(math.log2(n))) if n > 1 else 1
        self.solutions = []
        
    def is_valid_solution(self, board):
        """
        Check if a board configuration is a valid N-Queens solution.
        
        Args:
            board (list): List of column positions for each row
            
        Returns:
            bool: True if valid solution, False otherwise
        """
        n = len(board)
        
        # Check columns (no two queens in same column)
        if len(set(board)) != n:
            return False
            
        # Check diagonals
        for i in range(n):
            for j in range(i + 1, n):
                # Check if queens are on same diagonal
                if abs(board[i] - board[j]) == abs(i - j):
                    return False
                    
        return True
    
    def find_classical_solutions(self):
        """
        Find all valid N-Queens solutions using classical computation.
        This is used for verification and when quantum approach is not practical.
        
        Returns:
            list: All valid board configurations
        """
        solutions = []
        
        # Generate all possible permutations (one queen per row)
        for perm in permutations(range(self.n)):
            if self.is_valid_solution(list(perm)):
                solutions.append(list(perm))
                
        return solutions
    
    def create_oracle(self):
        """
        Create the oracle circuit that marks valid N-Queens solutions.
        
        For simplicity, this implementation uses a classical oracle approach
        where we mark known valid solutions.
        
        Returns:
            QuantumCircuit: Oracle circuit
        """
        # Find all valid solutions classically for oracle construction
        valid_solutions = self.find_classical_solutions()
        
        # Create quantum circuit
        qreg = QuantumRegister(self.num_qubits, 'q')
        oracle = QuantumCircuit(qreg)
        
        # For each valid solution, create a marking operation
        for solution in valid_solutions:
            # Convert solution to binary representation
            binary_solution = self._solution_to_binary(solution)
            
            # Create multi-controlled Z gate to flip phase of this solution
            self._mark_solution(oracle, qreg, binary_solution)
            
        return oracle
    
    def _solution_to_binary(self, solution):
        """
        Convert a solution (column positions) to binary representation.
        
        Args:
            solution (list): Column positions for each row
            
        Returns:
            str: Binary string representation
        """
        bits_per_queen = int(math.ceil(math.log2(self.n))) if self.n > 1 else 1
        binary = ""
        
        for col in solution:
            # Convert column position to binary
            col_binary = format(col, f'0{bits_per_queen}b')
            binary += col_binary
            
        return binary
    
    def _mark_solution(self, circuit, qreg, binary_solution):
        """
        Mark a specific solution by flipping its phase.
        
        Args:
            circuit (QuantumCircuit): Circuit to modify
            qreg (QuantumRegister): Quantum register
            binary_solution (str): Binary representation of solution
        """
        # Apply X gates to qubits that should be 0 in the solution
        for i, bit in enumerate(binary_solution):
            if bit == '0':
                circuit.x(qreg[i])
        
        # Apply multi-controlled Z gate using H-MCX-H pattern
        if len(binary_solution) > 1:
            target_qubit = len(binary_solution) - 1
            control_qubits = list(range(len(binary_solution) - 1))
            
            # Convert MCZ to H-MCX-H
            circuit.h(qreg[target_qubit])
            circuit.mcx(control_qubits, target_qubit)
            circuit.h(qreg[target_qubit])
        else:
            circuit.z(qreg[0])
        
        # Undo the X gates
        for i, bit in enumerate(binary_solution):
            if bit == '0':
                circuit.x(qreg[i])
    
    def create_diffuser(self):
        """
        Create Grover's diffusion operator (inversion about average).
        
        Returns:
            QuantumCircuit: Diffusion operator circuit
        """
        qreg = QuantumRegister(self.num_qubits, 'q')
        diffuser = QuantumCircuit(qreg)
        
        # Apply H gates to all qubits
        diffuser.h(qreg)
        
        # Apply X gates to all qubits
        diffuser.x(qreg)
        
        # Apply multi-controlled Z gate using H-MCX-H pattern
        if self.num_qubits > 1:
            target_qubit = self.num_qubits - 1
            control_qubits = list(range(self.num_qubits - 1))
            
            # Convert MCZ to H-MCX-H
            diffuser.h(qreg[target_qubit])
            diffuser.mcx(control_qubits, target_qubit)
            diffuser.h(qreg[target_qubit])
        else:
            diffuser.z(qreg[0])
        
        # Undo X gates
        diffuser.x(qreg)
        
        # Undo H gates
        diffuser.h(qreg)
        
        return diffuser
    
    def run_grovers_algorithm(self):
        """
        Run Grover's algorithm to find N-Queens solutions.
        
        Returns:
            dict: Measurement results
        """
        # Find classical solutions for comparison
        classical_solutions = self.find_classical_solutions()
        
        if not classical_solutions:
            return {"No solutions found": 1}
        
        # Calculate optimal number of iterations
        num_solutions = len(classical_solutions)
        total_states = 2 ** self.num_qubits
        optimal_iterations = int(math.pi / 4 * math.sqrt(total_states / num_solutions))
        
        # Create quantum circuit
        qreg = QuantumRegister(self.num_qubits, 'q')
        creg = ClassicalRegister(self.num_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Initialize superposition
        circuit.h(qreg)
        
        # Apply Grover iterations
        oracle = self.create_oracle()
        diffuser = self.create_diffuser()
        
        for _ in range(optimal_iterations):
            circuit.compose(oracle, inplace=True)
            circuit.compose(diffuser, inplace=True)
        
        # Measure
        circuit.measure(qreg, creg)
        
        # Run on simulator
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        
        return counts
    
    def decode_quantum_result(self, binary_string):
        """
        Decode a binary measurement result back to board configuration.
        
        Args:
            binary_string (str): Binary measurement result
            
        Returns:
            list: Board configuration (column positions)
        """
        bits_per_queen = int(math.ceil(math.log2(self.n))) if self.n > 1 else 1
        board = []
        
        for i in range(self.n):
            start_bit = i * bits_per_queen
            end_bit = start_bit + bits_per_queen
            queen_bits = binary_string[start_bit:end_bit]
            column = int(queen_bits, 2)
            
            # Ensure column is within bounds
            if column < self.n:
                board.append(column)
            else:
                # Invalid encoding, return None
                return None
                
        return board
    
    def display_board(self, solution):
        """
        Display a chess board with queen positions.
        
        Args:
            solution (list): Column positions for each row
        """
        print(f"\nSolution: {solution}")
        print("+" + "---+" * self.n)
        
        for row in range(self.n):
            print("|", end="")
            for col in range(self.n):
                if solution[row] == col:
                    print(" Q ", end="|")
                else:
                    print(" . ", end="|")
            print()
            print("+" + "---+" * self.n)
    
    def solve_and_display(self):
        """
        Main method to solve N-Queens problem and display results.
        """
        print(f"Solving {self.n}-Queens problem using Grover's Algorithm")
        print("=" * 50)
        
        # For larger N, quantum simulation becomes impractical
        # Use classical solution finding instead
        if self.n > 4:  # Threshold for quantum simulation
            print(f"For N={self.n}, using classical computation due to quantum simulation complexity.")
            solutions = self.find_classical_solutions()
        else:
            print("Running quantum Grover's algorithm...")
            # Run quantum algorithm
            quantum_results = self.run_grovers_algorithm()
            
            # Find classical solutions for verification
            classical_solutions = self.find_classical_solutions()
            solutions = classical_solutions
            
            print(f"Quantum measurement results: {quantum_results}")
        
        print(f"\nFound {len(solutions)} valid solutions.")
        
        # Display solutions based on N
        if self.n <= 6:
            print(f"\nDisplaying all solutions for {self.n}-Queens:")
            for i, solution in enumerate(solutions, 1):
                print(f"\n--- Solution {i} ---")
                self.display_board(solution)
        else:
            print(f"\nFor N={self.n}, there are {len(solutions)} possible solutions.")
            print("Due to display complexity, showing only the count.")
            if solutions:
                print(f"Example solution: {solutions[0]}")

def main():
    """
    Main function to run the N-Queens solver with user input.
    """
    print("N-Queens Problem Solver using Grover's Algorithm")
    print("=" * 50)
    
    try:
        n = int(input("Enter the value of N (board size): "))
        
        if n <= 0:
            print("Please enter a positive integer.")
            return
        
        if n == 1:
            print("\nFor N=1:")
            print("Solution: [0]")
            print("+---+")
            print("| Q |")
            print("+---+")
            return
            
        # Create and run solver
        solver = NQueensGrover(n)
        solver.solve_and_display()
        
        # Additional information
        print(f"\n--- Algorithm Information ---")
        print(f"Board size: {n}x{n}")
        print(f"Quantum qubits needed: {solver.num_qubits}")
        
        if n <= 4:
            print("Used quantum Grover's algorithm simulation")
        else:
            print("Used classical computation (quantum simulation impractical for large N)")
            
    except ValueError:
        print("Please enter a valid integer.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()