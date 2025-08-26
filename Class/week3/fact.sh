read -p "Enter a number: " n

if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "Error: Please enter a valid positive number"
    exit 1
fi

if [ "$n" -lt 0 ]; then
    echo "Error : Please enter only positive numbers"
    exit 1
fi

sum_n() {
    num=$1
    sum=0
    while [ "$num" -gt 0 ]; do
        sum=$((sum + num))
        num=$((num - 1))
    done
    echo $sum
}

sum=$(sum_n "$n")

echo "Sum of first $n natural numbers is: $sum"

factorial() {
    num=$1
    fact=1
    while [ "$num" -gt 0 ]; do
        fact=$((fact * num))
        num=$((num - 1))
    done
    echo $fact
}

fact=$(factorial "$n")

echo "Factorial of $n : $fact"
