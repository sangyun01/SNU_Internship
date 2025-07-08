module Tree_acc (
    input wire clk,
    input wire rst,
    input wire [255:0] x_in_flat,       // 16개 Q8.8을 1차원 벡터로
    output reg [15:0] mean_out
);

    wire [15:0] x_in [0:15];
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1)
            assign x_in[i] = x_in_flat[i*16 +: 16];
    endgenerate

    reg [16:0] stage1 [0:7];
    reg [17:0] stage2 [0:3];
    reg [18:0] stage3 [0:1];
    reg [19:0] stage4;

    integer j;

    always @(posedge clk) begin
        if (rst) begin
            mean_out <= 0;
        end else begin
            for (j = 0; j < 8; j = j + 1)
                stage1[j] <= x_in[2*j] + x_in[2*j+1];
            for (j = 0; j < 4; j = j + 1)
                stage2[j] <= stage1[2*j] + stage1[2*j+1];
            for (j = 0; j < 2; j = j + 1)
                stage3[j] <= stage2[2*j] + stage2[2*j+1];
            stage4 <= stage3[0] + stage3[1];
            mean_out <= stage4 >> 4;
        end
    end

endmodule

