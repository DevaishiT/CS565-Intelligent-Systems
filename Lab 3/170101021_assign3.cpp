#include <bits/stdc++.h>
using namespace std;

typedef long double ld;
typedef long long ll;
typedef pair<long double, long double> pld;


// Returns a transposed matrix for a given matrix
ld **takeTranspose (ld **matrix, ll row, ll column)
{
    ld **output = new ld *[column];

    for(ll i = 0; i < column; i++)
    {
        output[i] = new ld[row];
        for(ll j = 0; j < row; j++) 
            output[i][j] = matrix[j][i];
    }

    return output;
}

// Returns the dot product of a given matrix with a vector.
ld *dotProduct (ld **matrix, ld *vector, ll row, ll column)
{
    ld *output = new ld[row];

    // row-wise multiplication
    for(ll i = 0; i < row; i++)
    {
        output[i] = 0;
        for(ll j = 0; j < column; j++)
            output[i] += matrix[i][j] * vector[j];
    }

    return output;
}

// Performs softmax function over the given vector.
// (converting the output to a number between 0 and 1)
ld *softmax (ld *vector, ll length)
{
    ld *output = new ld[length];

    // Calculating the sum of all elements
    ld sum = 0;
    for(ll i = 0; i < length; i++)
    {
        output[i] = exp(vector[i]);
        sum += output[i];
    }

    for(ll i = 0; i < length; i++) output[i] /= sum;
    return output;
}

int main()
{
    ll vocab_size, num_dimension, num_iteration, num_pairs;
    ld learning_rate;

    // Taking in inputs from the text file
    cin >> vocab_size >> num_dimension >> learning_rate >> num_iteration >> num_pairs;

    vector<pld> word_pairs;
    ll iteration_id, input_word, output_word;

    // reading the input output pairs sequentially
    for (ll i = 0; i < num_pairs; i++)
    {
        cin >> iteration_id >> input_word >> output_word;
        word_pairs.push_back(make_pair(input_word-1, output_word-1));
    }

    ld **input_weight = new ld *[vocab_size];
    ld **hidden_weight = new ld *[num_dimension];


    // Initialising both input and hidden weight matrices
    for (ll i = 0; i < vocab_size; i++)
    {
        input_weight[i] = new ld[num_dimension];
        for (ll j = 0; j < num_dimension; j++)
        {
            input_weight[i][j] = 0.5;
        }
    }

    for (ll i = 0; i < num_dimension; i++)
    {
        hidden_weight[i] = new ld[vocab_size];
        for (ll j = 0; j < vocab_size; j++)
        {
            hidden_weight[i][j] = 0.5;
        }
    }

    for (ll i = 0; i < num_iteration; i++)
    {
        for (ll j = 0; j < num_pairs; j++)
        {
            input_word = word_pairs[j].first;
            output_word = word_pairs[j].second;

            ld positive_change = 0, negative_change = 0;

            ld *input_vector = new ld[vocab_size];
            for (ll k = 0; k < vocab_size; k++)
            {
                input_vector[k] = 0;
            }
            input_vector[input_word] = 1;

            // Calculating the value of the hidden layer elements
            ld **transpose = takeTranspose(input_weight, vocab_size, num_dimension);
            ld *hidden_vector = dotProduct(transpose, input_vector, num_dimension, vocab_size);

            // Calculating the value of the output vector
            transpose = takeTranspose(hidden_weight, num_dimension, vocab_size);
            ld *output_vector = dotProduct(transpose, hidden_vector, vocab_size, num_dimension);

            // Applying softmax function over the output vector
            ld *softmax_vector = softmax(output_vector, vocab_size);

            // Calculating the error encountered in the run
            ld *error_vector = new ld[vocab_size];
            for (ll k = 0; k < vocab_size; k++)
            {
                if (k == output_word) error_vector[k] = softmax_vector[k] - 1;
                else error_vector[k] = softmax_vector[k];
            }

            // dW - partial derivative of the error function wrt to weights.
            ld *dW = new ld[num_dimension];
            for (ll k = 0; k < num_dimension; k++)
                dW[k] = error_vector[output_word] * hidden_vector[k];

            // Calculating partial derivatives for the backpropogation step.
            ld *dH = new ld[num_dimension];
            for (ll k = 0; k < num_dimension; k++)
            {
                dH[k] = 0;
                for (ll l = 0; l < vocab_size; l++)
                    dH[k] += error_vector[l] * hidden_weight[k][l];
            }
            
            // Updating the weights and recording positive and negative changes 
            // in each hidden element's weight.
            for (ll k = 0; k < num_dimension; k++)
            {
                hidden_weight[k][output_word] -= learning_rate * dW[k];
                
                if (learning_rate * dW[k] > 0) negative_change++;
                else positive_change++;
            }

            // Updating the weights and recording positive and negative changes
            // in each input element's weight.
            for (ll l = 0; l < num_dimension; l++)
            {
                input_weight[input_word][l] -= learning_rate * dH[l];
                
                if(learning_rate * dH[l] > 0) negative_change++;
                else positive_change++;
            }

            // Printing the desired output at each step.
            cout << (i+1) << " " << (j+1) << " " << negative_change << " " << positive_change << endl;
        }
    }
}
