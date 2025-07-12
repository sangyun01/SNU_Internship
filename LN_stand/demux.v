module demux_bypass #(
    parameter N = 4,
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [N*WIDTH-1:0] din_vec,   // bypass 결과 벡터
    input  wire sel,                     // 1: bypass 결과 BRAM 저장, 0: bypass 결과 바로 출력

    output reg  [WIDTH-1:0] bram_data_in,  // BRAM에 쓸 데이터 (하위 16비트)
    output reg  [N*WIDTH-1:0] out_vec        // 최종 출력 벡터
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            bram_data_in <= 0;
            out_vec <= 0;
        end else if (en) begin
            if (sel) begin
                // bypass 결과를 BRAM에 저장 (하위 16비트만 저장)
                bram_data_in <= din_vec[WIDTH-1:0];
                out_vec <= 0;
            end else begin
                // bypass 결과를 바로 출력
                bram_data_in <= 0;
                out_vec <= din_vec;
            end
        end
    end

endmodule
