`timescale 1ns/1ps 

module unsigned_multiplier_gen #(parameter N = 4) (
    input [N-1:0] x,
    input [N-1:0] y,
    output [2*N-1:0] prod
);

wire sum [N-1:0][N-1:0];
wire carry [N-1:0][N-1:0];

genvar i, j;

generate for(i = 0; i < N; i = i + 1) begin : multiplier
    if(i == 0)
        for(j = 0; j < N; j = j + 1) begin
            assign sum [j][i] = x[j] & y[i];
            assign carry [j][i] = 0;
        end
    else
        for(j = 0; j < N; j = j + 1) begin
            if(j == 0)
                assign {carry[j][i], sum[j][i]} = sum[j+1][i-1] + (x[j]&y[i]);
            else if (j == N-1)
                assign {carry[j][i], sum[j][i]} = (x[j]&y[i]) + carry[N-1][i-1] + carry[j-1][i];
            else
                assign {carry[j][i], sum[j][i]} = (x[j]&y[i]) + sum[j+1][i-1] + carry[j-1][i];
        end
end endgenerate

generate for (i = 0; i < N; i = i + 1) begin: low_bit_multiplier
    assign prod[i] = sum[0][i];
end endgenerate 
generate for (i = 1; i < N; i = i + 1) begin: hifh_bit_multiplier
    assign prod[N-1+i] = sum[i][N-1];
end endgenerate 
assign prod[2*N-1] = carry[N-1][N-1];

endmodule
