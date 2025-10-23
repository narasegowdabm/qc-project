from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import numpy as np
from typing import List
import math

# Dictionary of known solution counts for N-Queens problems
KNOWN_SOLUTIONS = {
    1: 1,
    2: 0,
    3: 0,
    4: 2,
    5: 10,
    6: 4,
    7: 40,
    8: 92,
    9: 352,
    10: 724,
}

def n_queens_solver(n: int) -> List[str]:
    """
    Solve the N-Queens problem using Grover's algorithm.
    
    Args:
        n (int): Size of the chess board (n x n)
        
    Returns:
        List[str]: List of valid N-Queens solutions as board strings
    """
    if n > 6:
        return []  # Don't compute solutions for large boards
        
    num_qubits = n * n
    
    # Calculate optimal number of iterations based on board size
    if n == 4:
        num_iterations = 2  # Optimal for n=4
    elif n == 5:
        num_iterations = 8  # Further increased iterations for n=5
    else:
        # Estimate based on sqrt of search space / estimated solutions
        estimated_solutions = n  # rough estimate
        raw_iterations = int(math.pi/4 * math.sqrt(2**num_qubits / estimated_solutions))
        num_iterations = min(raw_iterations, 3)  # Cap at 3 for larger boards
    
    print(f"Using {num_iterations} Grover iterations for {n}x{n} board...")
    
    # Create quantum registers
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    circuit = QuantumCircuit(qr, cr)
    
    # Put all qubits in superposition
    circuit.h(qr)
    
    # Define simple oracle for 4-Queens
    # We'll use controlled-Z gates to mark invalid positions
    def oracle():
        # Mark invalid row combinations
        for row in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    circuit.cz(qr[row*n + i], qr[row*n + j])
        
        # Mark invalid column combinations
        for col in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    circuit.cz(qr[i*n + col], qr[j*n + col])
        
        # Mark invalid diagonal combinations
        for i in range(num_qubits):
            row1, col1 = i // n, i % n
            for j in range(i + 1, num_qubits):
                row2, col2 = j // n, j % n
                if abs(row1 - row2) == abs(col1 - col2):
                    circuit.cz(qr[i], qr[j])
    
    # Define diffusion operator
    def diffusion():
        circuit.h(qr)
        circuit.x(qr)
        
        # Apply controlled-Z to |11...1⟩ state
        circuit.h(qr[-1])
        circuit.x(qr[:-1])
        circuit.mcx(list(qr[:-1]), qr[-1])
        circuit.x(qr[:-1])
        circuit.h(qr[-1])
        
        circuit.x(qr)
        circuit.h(qr)
    
    # Apply calculated number of Grover iterations
    for _ in range(num_iterations):
        oracle()
        diffusion()
    
    # Measure all qubits
    circuit.measure(qr, cr)
    
    # Run on simulator with shots based on board size and available memory
    if n == 5:
        shots = 65536  # Adjusted for memory efficiency while maintaining good sampling
    else:
        shots = min(4096 * n, 16384)  # Reduced for larger boards to save memory
    print(f"Running quantum simulation with {shots} shots...")
    backend = Aer.get_backend('qasm_simulator')
    counts = backend.run(circuit, shots=shots).result().get_counts()
    
    def is_valid_solution(bitstring: str) -> bool:
        """Check if a bitstring represents a valid N-Queens solution."""
        # Convert bitstring to board
        board = [[0] * n for _ in range(n)]
        for i, bit in enumerate(bitstring):
            if bit == '1':
                row, col = i // n, i % n
                board[row][col] = 1
        
        # Check rows and columns
        for i in range(n):
            if sum(board[i]) != 1 or sum(board[j][i] for j in range(n)) != 1:
                return False
        
        # Check diagonals
        for i in range(n):
            for j in range(n):
                if board[i][j]:
                    for k in range(1, n):
                        # Check all diagonal directions
                        for di, dj in [(k,k), (k,-k), (-k,k), (-k,-k)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < n and board[ni][nj]:
                                return False
        return True
    
    def format_solution(bitstring: str) -> str:
        """Format a solution bitstring as a board string."""
        return '\n'.join(''.join('Q' if bitstring[i*n + j] == '1' else '.'
                                for j in range(n))
                        for i in range(n))
    
    # Collect valid solutions
    solutions = []
    for bitstring in counts:
        if is_valid_solution(bitstring):
            solution = format_solution(bitstring)
            if solution not in solutions:
                solutions.append(solution)
    
    return solutions

# Dictionary of known solution counts for N-Queens problems
KNOWN_SOLUTIONS = {
    1: 1,
    2: 0,
    3: 0,
    4: 2,
    5: 10,
    6: 4,
    7: 40,
    8: 92,
    9: 352,
    10: 724,
}

def main():
    try:
        # Get board size from user
        n = int(input("Enter the size of the board (e.g., 4 for 4x4): "))
        
        # Print total number of possible solutions
        if n in KNOWN_SOLUTIONS:
            print(f"\nThe {n}x{n} N-Queens puzzle has {KNOWN_SOLUTIONS[n]} possible solutions.")
        else:
            print(f"\nThe {n}x{n} N-Queens puzzle has multiple solutions (exact count > 724).")
        
        if n > 6:
            # print("\nFor boards larger than 6x6, only the solution count is shown.")
            print("Computing {n} solutions would require excessive quantum resources.")
            return
        
        print(f"\nSolving {n}x{n} N-Queens puzzle using Grover's algorithm...")
        print("(This might take a minute...)")
        if n < 4:
            print("Board size must be at least 4")
            return
        print(f"Solving {n}x{n} N-Queens puzzle using Grover's algorithm...")
        print("(This might take a minute...)")
        
        solutions = n_queens_solver(n)
        
        if solutions:
            print(f"\nFound {len(solutions)} solutions in this run:")
            for i, solution in enumerate(solutions, 1):
                print(f"\nSolution {i} of {KNOWN_SOLUTIONS[n]} possible solutions:")
                print(solution)
            
            if len(solutions) < KNOWN_SOLUTIONS[n]:
                print(f"\nNote: There are {KNOWN_SOLUTIONS[n]} total possible solutions for {n}x{n} board.")
                print("The quantum algorithm is probabilistic and may not find all solutions in one run.")
                print("Try running the program again to find more solutions.")
        else:
            print("\nNo valid solutions found in this run.")
            print(f"Note: The {n}x{n} N-Queens problem has {KNOWN_SOLUTIONS.get(n, 'multiple')} possible solutions.")
            print("The quantum algorithm is probabilistic and may not find all solutions in a single run.")
            print("Try running the program again.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()