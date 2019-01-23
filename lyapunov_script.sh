echo "s = 0.15"
for run in {1..10}
do
	python lyapunov.py 0.15
done

echo "s = 0.35"
for run in {1..10}
do
	python lyapunov.py 0.35
done

echo "s = 0.55"
for run in {1..10}
do
	python lyapunov.py 0.55
done

echo "s = 0.75"
for run in {1..10}
do
	python lyapunov.py 0.75
done

echo "s = 0.95"
for run in {1..10}
do
	python lyapunov.py 0.95
done

echo "s = 1.15"
for run in {1..10}
do
	python lyapunov.py 1.15
done