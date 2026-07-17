global matvec_scalar

section .text
matvec_scalar:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi           
    mov r13, rsi             
    mov r14, rdx              
    mov r15, rcx               
    ; r8 already holds cols

    xor rbx, rbx               

.row_loop:
    cmp rbx, r15
    jge .done

    mov rax, rbx
    imul rax, r8            
    lea r9, [r12 + rax*4]     

    xorps xmm0, xmm0            
    xor rcx, rcx                

.col_loop:
    cmp rcx, r8
    jge .row_done

    movss xmm1, [r9 + rcx*4]     
    movss xmm2, [r13 + rcx*4]    
    mulss xmm1, xmm2
    addss xmm0, xmm1

    inc rcx
    jmp .col_loop

.row_done:
    movss [r14 + rbx*4], xmm0
    inc rbx
    jmp .row_loop

.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
